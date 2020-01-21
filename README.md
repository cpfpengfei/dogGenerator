# dogGenerator
A variational autoencoder (VAE) that generates images of dogs (instead of cats)

The dog generator model was trained on the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) which consists of 20580 dog images of 120 different breeds.

## Model and Training
The src folder contains:
- The original VAE model (model.py)
- Useful helper functions (vaeHelpers.py)

The model is then trained in the notebook `DogGeneratorModel.ipynb`, with model weights stored in weights folder, and the model can be tested using the 2nd part of the same notebook.

## Generating Images
Individual dog images can be generated using the `genDogs.py` script by including the number of dogs we wish to generate.

## Current Project Status 
With current parameters, as it takes ~1.5h for 1 epoch to run, the model is still not trained yet. Especially due to the shear size of dataset. 
So oops, no Chihuahuas can be generated yet.

## Future Work
- This project's purpose is really to learn about VAE. Thereafter, I wish to explore GANs. 
- Possible extensions: Meme generator and embedding the dog/meme generator into a Telegram bot to allow on-demand dog/meme generation!


## Original Source and Credits:
This project is forked from [catGenerator](https://github.com/m0baxter/catGenerator) and the original project has helped me tremendously in learning about VAEs (while generating dog images).

