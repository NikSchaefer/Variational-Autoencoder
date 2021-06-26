import tensorflow as tf
from keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = fashion_mnist.load_data()


def process_dataset(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > 0.5, 1.0, 0.0).astype("float32")


train_images = process_dataset(x_train)
test_images = process_dataset(x_test)

train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(train_size)
    .batch(batch_size)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)
)


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

        self.latent_dim = 2

        self.encoder = Sequential(
            [
                layers.InputLayer(input_shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=3, strides=(2, 2), activation="relu"),
                layers.Conv2D(64, kernel_size=3, strides=(2, 2), activation="relu"),
                layers.Flatten(),
                layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        self.decoder = Sequential(
            [
                layers.InputLayer(input_shape=(self.latent_dim,)),
                layers.Dense(7 * 7 * 32, activation="relu"),
                layers.Reshape(target_shape=(7, 7, 32)),
                layers.Conv2DTranspose(
                    64, kernel_size=3, strides=2, padding="same", activation="relu"
                ),
                layers.Conv2DTranspose(
                    32, kernel_size=3, strides=2, padding="same", activation="relu"
                ),
                layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding="same"),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        return x_logit


optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 10

num_examples_to_generate = 16

model = VariationalAutoEncoder()


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    # computes probable difference in logit and label
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)

    # Monte Carlo Estimate
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


# callable graph function
@tf.function
def train_step(model, x, optimizer):
    # compute gradients for variables
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)

    # Apply calculated gradients to trainable variables
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def save_sample_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.savefig("epoch_images/image_at_epoch_{:04d}.png".format(epoch))


for test_batch in test_dataset.take(1):
    sample_batch = test_batch[0:num_examples_to_generate]


for epoch in range(1, epochs + 1):
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    loss = loss.result()
    print("Epoch: {}, Loss(ELBO): {}".format(epoch, loss))
    save_sample_images(model, epoch, sample_batch)

