# catGenerator
A variational autoencoder that generates images of cats or dogs.

The cat moddel was trained on cropped versions of the data from the [Kaggle cat dataset](https://www.kaggle.com/crawford/cat-dataset) while the dog images are cropped versions of the data in the [Kaggle dog breed challenge](https://www.kaggle.com/c/dog-breed-identification). The cropping was performed using a modified version of the [TensorFlow object detection example](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Training

The models can be trained by running `trainModel.ipynb` with appropriate parameters set (path to the image data). The weights that result from training the model with with default parameters for both cats and dogs are available in the `./weights/` directory.

## Testing

The can be tested by running `testModel.ipnb`. The reconstructions produced are reasonable, however the generated images suffer from a few problems. These issues could perhaps be alleviated by tunning the architecture/hyperparameters. A better solution might be to crop the images to faces. This would hopefully allow the model to focus more on one aspect of the animals rather than trying to recreate entire bodies that can be in any position/size.