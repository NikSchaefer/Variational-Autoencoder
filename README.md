# Variational Autoencoder

Variational Autoencoder based on the fashion_mnist dataset

## How it works

the encoder is a Sequential model with 2 Convolutional layers and then a Dense. The decoder works to undo the encoder and has a Dense layer first and then contains Conv2dTranspose layers to undo the Conv2d Layers.

We calculate loss by using the Monte Carlo estimate and use sigmoid cross entropy to determine the probable difference between the image and reconstruction. The model uses an Adam optimizer to apply gradients to the model.

## Data

Data is from the fashion_mnist dataset on [tensorflow datasets](https://github.com/tensorflow/datasets)

```py
from keras.datasets import fashion_mnist
(x_train, _), (x_test, _) = fashion_mnist.load_data()
```

## Installation

Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
