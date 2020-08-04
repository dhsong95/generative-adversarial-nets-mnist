from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


class GANNetwork:
    def __init__(self):
        self.image_row = 28
        self.imgae_col = 28
        self.imgae_channel = 1
        self.image_shape = (self.image_row, self.imgae_col, self.imgae_channel)

        self.noise_shape = (100,)

        self.optimizer = keras.optimizers.Adam()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy']
        )

        self.generator = self.build_generator()
        self.generator.compile(
            optimizer=self.optimizer, loss='binary_crossentropy'
        )

        z = keras.Input(shape=self.noise_shape)
        generated_image = self.generator(z)

        self.discriminator.trainable = False
        validity = self.discriminator(generated_image)

        self.combined = keras.Model(z, validity)
        self.combined.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def build_generator(self):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(units=256, input_shape=self.noise_shape)
        )
        model.add(
            keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            keras.layers.BatchNormalization(momentum=0.8)
        )

        model.add(
            keras.layers.Dense(units=256)
        )
        model.add(
            keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            keras.layers.BatchNormalization(momentum=0.8)
        )

        model.add(
            keras.layers.Dense(units=1024)
        )
        model.add(
            keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            keras.layers.BatchNormalization(momentum=0.8)
        )

        model.add(
            keras.layers.Dense(units=np.prod(self.image_shape), activation='tanh')
        )
        model.add(
            keras.layers.Reshape(target_shape=self.image_shape)
        )

        model.summary()

        noise = keras.Input(shape=self.noise_shape)
        generated_image = model(noise)

        return keras.Model(noise, generated_image)

    def build_discriminator(self):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Flatten(input_shape=self.image_shape)
        )
        model.add(
            keras.layers.Dense(units=512)
        )
        model.add(
            keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            keras.layers.Dense(units=256)
        )
        model.add(
            keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            keras.layers.Dense(units=1, activation='sigmoid')
        )

        image = keras.Input(self.image_shape)
        validity = model(image)

        return keras.Model(image, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        (X_train, _), (_, _) = mnist.load_data()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=-1)

        half_batch_size = batch_size // 2

        for epoch in range(epochs):
            indices = np.random.randint(0, X_train.shape[0], half_batch_size)
            images = X_train[indices]

            noise = np.random.normal(0, 1, (half_batch_size, 100))
            generated_images = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(images, np.ones((half_batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_images, np.zeros((half_batch_size, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            validity = np.ones((batch_size, 1))
            g_loss = self.combined.train_on_batch(noise, validity)


            print('{:>8d}\t[D loss: {:.4f}, acc: {:.2%}]\t[G loss: {:.4f}]'.format(epoch, d_loss[0], d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_images(epoch)

    def save_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('images/mnist_{}.png'.format(epoch))
        plt.close()


if __name__ == "__main__":
    gan = GANNetwork()
    gan.train(epochs=30000, batch_size=32, save_interval=200)
