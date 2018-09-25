
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.model import genModel

def genCats( n ):
    """Generates n cat images."""

    encCat, decCat, catVAE = genModel( imgSize = 128, codeSize = 256, filters = 128 )
    catVAE.load_weights( "weights/catGen.hdf5" )

    codes = np.random.normal( size = (n, 256) )
    cats = decCat.predict( codes )

    return cats

def saveCat( cat, label ):

    fig = plt.figure( figsize = (1, 1), dpi = 128 )

    plt.imshow( cat, vmin = 0, vmax = 1  )
    plt.axis('off')

    fig.savefig( "cat-{0}.png".format(label), dpi = 128 )

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Invalid command line arguments."

    try:
        n = int(sys.argv[1])

    except ValueError:
        print( "Command-line arguement must be an integer." )

    else:
        cats = genCats( n )
        
        for i in range( n ):
            saveCat( cats[i], str(i) )

