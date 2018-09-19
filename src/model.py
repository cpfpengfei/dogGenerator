
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,\
                         Conv2DTranspose, Flatten, Reshape, Lambda, ELU, BatchNormalization, Activation
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from keras.optimizers import Adam, SGD
import keras.backend as kb


def sampling( args ):
    """Samples from a normal distribution with mean args[0] and log-variance args[1]."""

    mean, logSigma = args
    batch = kb.shape( mean )[0]
    dim   = kb.int_shape( mean )[1]

    epsilon = kb.random_normal( shape = (batch, dim) )

    return mean + kb.exp(0.5 * logSigma ) * epsilon

def genModel( codeSize  = 2048, imgSize = 256, filters = 64, colour = 3, lossType = "mse" ):
    """Generates the VAE model."""

    #Build  the network:
    factor = imgSize * imgSize

    inputs = Input( shape = ( imgSize, imgSize, colour,) )
    
    conv1 = Conv2D( filters//8, 3, strides = 1, padding = 'same', use_bias = None,
                    data_format = "channels_last")( inputs )
    bn1   = ELU()(BatchNormalization()(conv1))
    pool1 = MaxPooling2D( data_format = "channels_last" )( bn1 )
    
    conv2 = Conv2D( filters//4, 3, strides = 1, padding = 'same', use_bias = None,
                    data_format = "channels_last" )( pool1 )
    bn2   = ELU()(BatchNormalization()(conv2))
    pool2 = MaxPooling2D( data_format = "channels_last" )( bn2 )
    
    conv3 = Conv2D( filters//2, 3, strides = 1, padding = 'same', use_bias = None,
                    data_format = "channels_last" )( pool2 )
    bn3   = ELU()(BatchNormalization()(conv3))
    pool3 = MaxPooling2D( data_format = "channels_last" )( bn3 )
    
    conv4 = Conv2D( filters, 3, strides = 1, padding = 'same',
                    data_format = "channels_last" )( pool3 )
    bn4   = ELU()(BatchNormalization()(conv4))
    pool4 = MaxPooling2D( data_format = "channels_last" )( bn4 )
    
    flat  = Flatten()( pool4 )
    
    dense1 = Dense( codeSize, use_bias = False )( flat )
    bnd1 = ELU()(BatchNormalization()(dense1))
    
    mean     = Dense( codeSize )( bnd1 )
    logSigma = Dense( codeSize )( bnd1 )
    
    encoding = Lambda(sampling, output_shape = ( codeSize,) )( [mean, logSigma] )
    
    decoderInput = Input( shape = (codeSize, ) )
    
    dense2 = Dense( codeSize, use_bias = False )( decoderInput )
    bnd2 = ELU()(BatchNormalization()(dense2))
    
    dense3 = Dense( filters * 8 * 8, use_bias = False )( bnd2 )
    bn6   = ELU()(BatchNormalization()(dense3))
    
    unflatten = Reshape( (8, 8, filters) )(bn6 )
    
    unpool2 = UpSampling2D( data_format = "channels_last" )( unflatten )
    deconv2 = Conv2DTranspose( filters//2, 3, strides = 1, padding = 'same',
                               data_format = "channels_last" )( unpool2 )
    bn8     = ELU()(BatchNormalization()( deconv2 ))
    
    unpool3 = UpSampling2D( data_format = "channels_last" )( bn8 )
    deconv3 = Conv2DTranspose( filters//4, 3, strides = 1, padding = 'same',
                               use_bias = None, data_format = "channels_last" )( unpool3 )
    bn9     = ELU()(BatchNormalization()( deconv3 ))

    unpool4 = UpSampling2D( data_format = "channels_last" )( bn9 )
    deconv4 = Conv2DTranspose( filters//8, 3, strides = 1, padding = 'same', use_bias = None,
                               data_format = "channels_last" )( unpool4 )
    bn10    = ELU()(BatchNormalization()( deconv4 ))

    unpool5  = UpSampling2D( data_format = "channels_last" )( bn10 )
    decoding = Conv2DTranspose( colour, 3, strides = 1, padding = 'same',
                                data_format = "channels_last", activation = "sigmoid" )( unpool5 )

    #Define the models:
    encoder = Model( inputs, [mean, logSigma, encoding], name = 'encoder')
    decoder = Model( decoderInput, decoding, name = "decoder")
    
    fullPass = decoder( encoder(inputs)[2] )
    
    VAE = Model( inputs, fullPass, name = 'vae' )

    if ( lossType == "mse" ):
        reconLoss = factor * mse( kb.flatten( inputs ), kb.flatten( fullPass ) )

    elif ( lossType == "crossEnt"):
        reconLoss = factor * binary_crossentropy( kb.flatten( inputs ), kb.flatten( fullPass ) )
        
    klLoss = -0.5 * kb.sum(1 + logSigma - kb.square(mean) - kb.exp(logSigma), axis = -1 )
    loss = kb.mean( reconLoss + klLoss )
    
    opt = Adam() #Adam( lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1.0, decay = 0.1 )
    
    VAE.add_loss( loss )
    VAE.compile( optimizer = opt, loss = None )

    return ( encoder, decoder, VAE )

