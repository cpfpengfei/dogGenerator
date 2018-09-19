
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

def convBlock(x, filters):

    conv = Conv2D( filters, 3, strides = 1, padding = 'same', use_bias = None,
                    data_format = "channels_last")( x )
    bn   = ELU()(BatchNormalization()(conv))
    pool = MaxPooling2D( data_format = "channels_last" )( bn )

    return pool

def deconvBlock(x, filters):

    unpool = UpSampling2D( data_format = "channels_last" )( x )
    deconv = Conv2DTranspose( filters, 3, strides = 1, padding = 'same',
                               data_format = "channels_last" )( unpool )
    bn     = ELU()(BatchNormalization()( deconv ))

    return bn

def genModel( codeSize  = 2048, imgSize = 256, filters = 64, lossType = "mse" ):
    """Generates the VAE model."""

    #Build  the network:
    factor = imgSize * imgSize

    inputs = Input( shape = ( imgSize, imgSize, 3,) )

    conv1 = convBlock( inputs, filters//8 )
    conv2 = convBlock( conv1,  filters//4 )
    conv3 = convBlock( conv2,  filters//2 )
    conv4 = convBlock( conv3,  filters )

    flat  = Flatten()( conv4 )

    dense1 = Dense( codeSize, use_bias = False )( flat )
    bnd1 = ELU()(BatchNormalization()(dense1))

    mean     = Dense( codeSize )( bnd1 )
    logSigma = Dense( codeSize )( bnd1 )

    encoding = Lambda(sampling, output_shape = ( codeSize,) )( [mean, logSigma] )

    decoderInput = Input( shape = (codeSize, ) )

    dense2 = Dense( codeSize, use_bias = False )( decoderInput )
    bnd2 = ELU()(BatchNormalization()(dense2))

    dense3 = Dense( filters * 8 * 8, use_bias = False )( bnd2 )
    bnd3   = ELU()(BatchNormalization()(dense3))

    unflatten = Reshape( (8, 8, filters) )(bnd3 )

    deconv1  = deconvBlock( unflatten, filters//2 )
    deconv2  = deconvBlock( deconv1,   filters//4 )
    deconv3  = deconvBlock( deconv2,   filters//8 )
    decoding = deconvBlock( deconv3,   3 )

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
