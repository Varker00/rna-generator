import os
import time

import numpy as np
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, ReLU, LeakyReLU

from data_funcs import inverse_scale_data, scale_data, gradient_penalty, compute_gradient_norm, calculate_fid


class GAN(Model):
    def __init__(self, data, latent_dim=64):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim

        self.original_data = data
        self.original_min = data.min().min()
        self.original_max = data.max().max()
        self.data = scale_data(data)

        self.data_shape = data.shape[1]

        self.build_generator()
        self.build_discriminator()
        self.build_combined()

    def build_generator(self, num_layers=4, num_neurons=256):
        model = Sequential()

        model.add(Dense(num_neurons, input_dim=self.latent_dim))
        model.add(BatchNormalization())
        model.add(ReLU())

        for _ in range(num_layers - 1):
            model.add(Dense(num_neurons))
            model.add(BatchNormalization())
            model.add(ReLU())

        model.add(Dense(self.data_shape, activation='sigmoid'))

        noise = Input(shape=(self.latent_dim,))
        generated_data = model(noise)

        self.generator = Model(noise, generated_data)
        return Model(noise, generated_data)

    def build_discriminator(self, num_layers=4, num_neurons=256):
        model = Sequential()

        model.add(Dense(num_neurons, input_dim=self.data_shape))
        model.add(LeakyReLU())

        for _ in range(num_layers - 1):
            model.add(Dense(num_neurons))
            model.add(LeakyReLU())

        model.add(Dense(1, activation='sigmoid'))

        data = Input(shape=(self.data_shape,))
        validity = model(data)

        self.discriminator = Model(data, validity)
        return Model(data, validity)

    def build_combined(self):
        z = Input(shape=(self.latent_dim,))
        generated_data = self.generator(z)

        self.discriminator.trainable = False
        validity = self.discriminator(generated_data)

        self.combined = Model(z, validity)

    def compile(self, loss, opt):
        self.generator.compile(loss=loss, optimizer=opt)
        self.discriminator.compile(loss=loss, optimizer=opt)
        self.combined.compile(loss=loss, optimizer=opt)

    def generate_data(self, n_samples):
        noise = np.random.normal(0, 1, size=(n_samples, self.latent_dim))
        generated_data = self.generator.predict(noise)
        generated_data = inverse_scale_data(generated_data, self.original_min, self.original_max)
        return generated_data

    def train(self, epochs, batch_size=128, n_critic=1, lambda_gp=10.0, save_interval=50):
        X_train = self.data.values

        # Initialize lists to store loss values
        d_loss_list = []
        g_loss_list = []
        gradient_norm_list = []
        fid_score_list = []
        gradient_norm = 0
        fid_score = 0
        best_fid_score = 1000000
        patience_stop = 15
        patience_decay = 4
        # Pętla po epokach
        for epoch in range(epochs):
            start = time.time()
            for _ in range(n_critic):
                # ---------------------
                #  Trenowanie dyskryminatora
                # ---------------------

                self.discriminator.trainable = True

                # Wybieranie losowych próbek
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_data = X_train[idx]

                # Generowanie nowego szumu
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generowanie nowych danych
                generated_data = self.generator.predict(noise)

                # Trenowanie dyskryminatora
                d_loss_real = self.discriminator.train_on_batch(real_data, -np.ones((batch_size, 1), dtype=np.float32))
                d_loss_fake = self.discriminator.train_on_batch(generated_data,
                                                                np.ones((batch_size, 1), dtype=np.float32))
                d_loss = d_loss_fake - d_loss_real + lambda_gp * gradient_penalty(real_data, generated_data,
                                                                                  self.discriminator)

                gradient_norm = compute_gradient_norm(self.discriminator, real_data, generated_data)
                gradient_norm_list.append(gradient_norm)

            # ---------------------
            #  Trenowanie generatora
            # ---------------------

            self.discriminator.trainable = False

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Chcemy, aby dyskryminator uznał wygenerowane dane za prawdziwe
            valid_y = np.array([1] * batch_size, dtype=np.float32)

            # Trenowanie generatora
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Append loss values to lists
            d_loss_list.append(d_loss)
            g_loss_list.append(g_loss)

            # Zapisywanie postępów
            if epoch % save_interval == 0:
                noise = np.random.normal(0, 1, size=(self.original_data.shape[0], self.latent_dim))
                generated_data = self.generator.predict(noise)
                generated_data = inverse_scale_data(generated_data, self.original_min, self.original_max)
                fid_score = calculate_fid(self.original_data, generated_data)
                fid_score_list.append(fid_score)
                print("%d [D loss: %f] [G loss: %f] [D gradient norm: %f] [FID score: %f]" % (
                    epoch, d_loss, g_loss, gradient_norm, fid_score))
                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                if fid_score < best_fid_score:
                    best_fid_score = fid_score
                    # best_generator = self.generator
                    patience_stop = 15
                    patience_decay = 4
                else:
                    patience_stop -= 1
                    patience_decay -= 1
                if patience_stop == 0:
                    print('Early stopping')
                    break
                if patience_decay == 0:
                    print('Decaying learning rate... New learning rate: ' + str(
                        (self.discriminator.optimizer.learning_rate / 2).numpy()))
                    patience_decay = 4
                    self.combined.optimizer.learning_rate = self.combined.optimizer.learning_rate / 2
                    self.discriminator.optimizer.learning_rate = self.discriminator.optimizer.learning_rate / 2

        return d_loss_list, g_loss_list, gradient_norm_list, fid_score_list  # , best_generator

    def save_model(self, save_path="./"):
        self.generator.save(os.path.join(save_path, 'generator.h5'))
        self.discriminator.save(os.path.join(save_path, 'discriminator.h5'))
        self.combined.save(os.path.join(save_path, 'combined.h5'))
