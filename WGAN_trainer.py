import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import TensorBoard
from utils import compute_gradient_norm, calculate_fd, inverse_scale_data, remove_checkpoints, wasserstein_loss

EPOCHS = 1000

class WGANTrainer:
    def __init__(self, data_processor, generator, discriminator, latent_dim, g_lr, d_lr, opt, id, n_critic):
        self.latent_dim = latent_dim
        self.opt = opt
        self.n_critic = n_critic
        self.dataprocessor = data_processor
        self.generator = generator
        self.discriminator = discriminator
        self.combined = None
        self.combined_optimizer = None
        self.discriminator_optimizer = None
        self.compile_models(g_lr, d_lr, opt)
        self.d_loss_list, self.g_loss_list, self.fd_score_list, self.gradient_norm_list = [], [], [], []
        self.log_dir = f'logs/WGAN/{self.dataprocessor.data.shape[1]}/{EPOCHS}/{id}'
        self.tensorboard = TensorBoard(log_dir=self.log_dir)
        self.tensorboard.set_model(self.combined)
        self.last_epoch = 0

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.combined_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    def save_checkpoint(self, file_path):
        self.checkpoint.save(file_path)
    
    def load_checkpoint(self, file_path):
        self.checkpoint.restore(tf.train.latest_checkpoint(file_path))

        
    def compile_models(self, g_lr, d_lr, opt):
        if opt == 'RMSprop':
            self.discriminator_optimizer = RMSprop(learning_rate=d_lr)
        elif opt == 'Adam':
            self.discriminator_optimizer = Adam(learning_rate=d_lr)
        self.discriminator.compile(loss=wasserstein_loss, optimizer=self.discriminator_optimizer)

        z = Input(shape=(self.latent_dim,))
        generated_data = self.generator(z)

        self.discriminator.trainable = False
        validity = self.discriminator(generated_data)

        self.combined = Model(z, validity)
        if opt == 'RMSprop':
            self.combined_optimizer = RMSprop(learning_rate=g_lr)
        elif opt == 'Adam':
            self.combined_optimizer = Adam(learning_rate=g_lr)
        self.combined.compile(loss=wasserstein_loss, optimizer=self.combined_optimizer)
    
    def train(self, data, epochs, batch_size, save_interval, checkpoint_dir):
        X_train = data.values
        
        best_fd_score = 10000000
        patience_stop, patience_decay = 10, 3

        remove_checkpoints(checkpoint_dir)
        self.checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "ckpt"))

        for epoch in range(self.last_epoch, self.last_epoch+epochs):
            start = time.time()
            d_loss = 0
            gradient_norm = 0
            for _ in range(self.n_critic):
                # Train Discriminator
                self.discriminator.trainable = True
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_data = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                generated_data = self.generator.predict(noise, verbose=0)
                d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((batch_size, 1), dtype=np.float32))
                d_loss_fake = self.discriminator.train_on_batch(generated_data, -np.ones((batch_size, 1), dtype=np.float32))
                d_loss = d_loss_real - d_loss_fake

                gradient_norm = compute_gradient_norm(self.discriminator, real_data, generated_data)

            # Train Generator
            self.discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_y = np.array([1] * batch_size, dtype=np.float32)
            g_loss = self.combined.train_on_batch(noise, valid_y)

            self.d_loss_list.append(d_loss)
            self.gradient_norm_list.append(gradient_norm)
            self.g_loss_list.append(g_loss)

            logs = {'d_loss': d_loss, 'g_loss': g_loss, 'gradient_norm': gradient_norm}
            self.tensorboard.on_epoch_end(epoch, logs)

            self.last_epoch = epoch

            # Save Progress
            if epoch % save_interval == 0:
                original_data = self.dataprocessor.get_original_data()
                original_min, original_max = original_data.min().min(), original_data.max().max()

                generated_data = self.generate_data(data.shape[0])
                generated_data = inverse_scale_data(generated_data, original_min, original_max)
                fd_score = calculate_fd(original_data, generated_data)
                self.fd_score_list.append(fd_score)
                logs = {'fd_score': fd_score}
                self.tensorboard.on_epoch_end(epoch, logs)
                self.print_progress(epoch, d_loss, g_loss, gradient_norm, fd_score, time.time() - start)

                if fd_score < best_fd_score:
                    best_fd_score = fd_score
                    remove_checkpoints(checkpoint_dir)
                    self.checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "ckpt"))
                    patience_stop, patience_decay = 10, 3
                    print(f"Checkpoint saved at epoch {epoch} with FD score: {fd_score}")
                else:
                    patience_stop -= 1
                    patience_decay -= 1

                if patience_stop == 0 or patience_decay == 0:
                    self.adjust_learning_rate(patience_decay)
                    patience_decay = 3
                    if patience_stop == 0: 
                        print("Early stopping...")
                        break
                    
        self.load_checkpoint(checkpoint_dir)
        remove_checkpoints(checkpoint_dir)
        return self.d_loss_list, self.g_loss_list, self.gradient_norm_list, self.fd_score_list
    
    def print_progress(self, epoch, d_loss, g_loss, gradient_norm, fd_score, elapsed_time):
        print(f"{epoch} [D loss: {d_loss:.6f}] [G loss: {g_loss:.6f}] [D gradient norm: {gradient_norm:.6f}] [FD score: {fd_score:.6f}]")
        print(f"Time for epoch {epoch + 1} is {elapsed_time:.2f} sec")

    def adjust_learning_rate(self, patience_decay):
        if patience_decay == 0:
            new_lr = self.discriminator_optimizer.learning_rate.numpy() / 2
            print(f"Decaying learning rate... New learning rate: {new_lr}")
            self.combined_optimizer.learning_rate.assign(new_lr)
            self.discriminator_optimizer.learning_rate.assign(new_lr)

    def generate_data(self, n_samples):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.generator.predict(noise)
    
    def save_models(self, file_path):
        self.generator.save(file_path + 'generator.h5')
        self.discriminator.save(file_path + 'discriminator.h5')
