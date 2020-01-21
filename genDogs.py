# this is used to generate n number of dog images based on the autoencoder 

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.model import genModel

def genDogs( n ):
    """Generates n dog images."""

    encDog, decDog, dogVAE = genModel( imgSize = 128, codeSize = 256, filters = 128 )
    dogVAE.load_weights( "weights/dogGen.hdf5" )

    codes = np.random.normal( size = (n, 256) )
    dogs = decDog.predict( codes )

    return dogs

def saveDog( dog, label ):

    fig = plt.figure( figsize = (1, 1), dpi = 128 )

    plt.imshow( dog, vmin = 0, vmax = 1  )
    plt.axis('off')

    fig.savefig( "dog-{0}.png".format(label), dpi = 128 )

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Invalid command line arguments."

    try:
        n = int(sys.argv[1])

    except ValueError:
        print( "Command-line arguement must be an integer." )

    else:
        dogs = genDogs( n )
        
        for i in range( n ):
            saveDog( dogs[i], str(i) )

