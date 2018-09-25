
import matplotlib
import matplotlib.pyplot as plt
import src.imageTrans as it
import numpy as np
from glob import glob
from keras.preprocessing import image


def readSavedFiles( path ):

    files = []

    with open( path, "r" ) as readFile:
        for line in readFile:
            files.append( line.strip() )

    return np.array( files )

def writeFilesList( path, files ):
    
    with open( path, "w") as outFile:
        for f in files:
            outFile.write( f + "\n" )

def genData( files, size = 200 ):
    """Generates a set of training data and its associated labels."""

    X = np.zeros( (len(files), size, size, 3) )

    i = 0

    for f in files:
        img  = image.load_img( f, target_size = (size, size), grayscale = False )
        img  = image.img_to_array(img)/255
        X[i] = img

        i += 1

    return X, None

def genBatch( files, batchSize, imgSize = 200, rnd = False ):
    """Generator of mini batches for training."""

    while (True):
        inds = np.random.permutation( len(files) )

        for start in range(0, len(files) - 1, batchSize):

            X, _ = genData( files[ inds[start : start + batchSize] ], size = imgSize )

            if (rnd):
                #brightness:
                if ( np.random.rand() < 0.5 ):
                    X = it.adjustBrightness( X, np.random.uniform(-0.1, 0.1) )

                #mirror flip:
                if ( np.random.rand() < 0.5 ):
                    X = it.mirrorImages( X, 0 )

                #rotate:
                if ( np.random.rand() < 0.5 ):
                    X = it.rotateImages( X, np.random.uniform(-np.pi/18, np.pi/18) )

            yield X, None

def plotLosses( losses ):
    """Plots training loss as a fucntion of epoch."""

    fig = plt.figure(1, figsize = (18,10))
    plt.plot( range(1, len(losses["loss"]) + 1), losses["loss"], "b-",
              linewidth = 3, label = "$\mathrm{training}$")
    plt.plot( range(1, len(losses["val_loss"]) + 1), losses["val_loss"], "g-",
              linewidth = 3, label = "$\mathrm{validation}$")
    plt.ylabel("$\mathrm{Loss}$")
    plt.xlabel("$\mathrm{Epoch}$")
    plt.legend( loc = "best" )
    
    plt.yscale( "log" )

    plt.show()
    #fig.savefig( "lossPlot.eps", format = 'eps', dpi = 20000, bbox_inches = "tight" )

    return

def plotGrid( data, title, size = (10, 10) ):
    """Plots a grid of images. Assumes that len(data) is a perfect square."""

    m = int(np.sqrt( len(data)) )
    f, axarr = plt.subplots(m, m, figsize = size )

    k = 0

    f.suptitle( title )

    for i in range(m):
        for j in range(m):

            axarr[i,j].imshow( data[k,:,:,:], vmin = 0, vmax = 1  )
            axarr[i,j].get_xaxis().set_ticks([])
            axarr[i,j].get_yaxis().set_ticks([])

            k += 1

    f.tight_layout( pad = 0.5 )
    plt.show()

