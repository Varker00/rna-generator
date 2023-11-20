from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Dropout, Input

class Generator:
    def __init__(self, latent_dim, data_shape, num_layers, num_neurons):
        self.model = self._build_model(latent_dim, data_shape, num_layers, num_neurons)

    def _build_model(self, latent_dim, data_shape, num_layers, num_neurons):
        model = Sequential()
        model.add(Dense(num_neurons, input_dim=latent_dim))
        model.add(BatchNormalization())
        model.add(ReLU())

        for _ in range(num_layers - 1):
            model.add(Dense(num_neurons))
            model.add(BatchNormalization())
            model.add(ReLU())

        model.add(Dense(data_shape, activation='tanh'))
        noise = Input(shape=(latent_dim,))
        generated_data = model(noise)

        return Model(noise, generated_data)

    def get_model(self):
        return self.model


class Discriminator:
    def __init__(self, data_shape, num_layers, num_neurons, dropout):
        self.model = self._build_model(data_shape, num_layers, num_neurons, dropout)

    def _build_model(self, data_shape, num_layers, num_neurons, dropout):
        model = Sequential()
        model.add(Dense(num_neurons, input_dim=data_shape))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))

        for _ in range(num_layers - 1):
            model.add(Dense(num_neurons))
            model.add(LeakyReLU())
            model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid'))
        data = Input(shape=(data_shape,))
        validity = model(data)

        return Model(data, validity)

    def get_model(self):
        return self.model
