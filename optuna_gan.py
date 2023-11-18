import json
import os
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import time
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
import optuna

from Trainer import Trainer
from data_funcs import wasserstein_loss, calculate_fid


# Load the data
data_path = 'original_data/GSE158508_normalized_counts.tsv'
data = pd.read_csv(data_path, sep='\t', index_col=0)

data_shape = data.shape[1]  # 69 columns

# Save columns names for later use
col_names = data.columns.values


def objective(trial):
    # Sugerowanie hiperparametrów
    num_layers_generator = trial.suggest_int('num_layers_generator', 2, 6)
    num_layers_discriminator = trial.suggest_int('num_layers_discriminator', 2, 6)
    neurons_per_layer_generator = trial.suggest_categorical('neurons_per_layer_generator', [64, 128, 256, 512])
    neurons_per_layer_discriminator = trial.suggest_categorical('neurons_per_layer_discriminator', [64, 128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    n_critic = trial.suggest_int('n_critic', 1, 5)
    lambda_gp = trial.suggest_float('lambda_gp', 1e-2, 1e2, log=True)
    latent_dim = trial.suggest_int('latent_dim', 32, 128)
    batch_size = 32
    optimizer = trial.suggest_categorical('optimizer', ['RMSprop', 'Adam'])
    if optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    trainer = Trainer(data, latent_dim)

    trainer.build_discriminator(num_layers=num_layers_discriminator, num_neurons=neurons_per_layer_discriminator)
    trainer.discriminator.compile(loss=wasserstein_loss, optimizer=opt)

    trainer.build_generator(num_layers=num_layers_generator, num_neurons=neurons_per_layer_generator)

    trainer.build_combined()
    trainer.combined.compile(loss=wasserstein_loss, optimizer=opt)

    trainer.train(epochs=50, batch_size=batch_size, n_critic=n_critic, lambda_gp=lambda_gp, save_interval=100)

    synthetic_data = trainer.generate_data(n_samples=57736)

    df = pd.DataFrame(synthetic_data)
    df.columns = col_names
    df.to_csv('synthetic_data/generated_data.tsv', sep='\t', index=False, header=True)
    df = pd.read_csv('synthetic_data/generated_data.tsv', sep='\t')
    synthetic_data = df

    # Obliczanie metryki, która ma być optymalizowana
    fid_score = calculate_fid(trainer.original_data, synthetic_data)

    return fid_score


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)

print('Best trial:')
trial = study.best_trial
print(' Value: ', trial.value)
print(' Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))


# save best model params to json
current_time = time.time()
current_struct_time = time.localtime(current_time)
save_dir_path = os.path.join('generation_results', time.strftime("%Y-%m-%d--%H-%M", current_struct_time))

if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

model_params_file = os.path.join(save_dir_path, 'model_params.json')

with open(model_params_file, 'w') as json_file:
    json.dump(trial.params, json_file, indent=4)
    print(f"JSON data has been saved to {model_params_file}")


# train the best model again and save it
latent_dim = trial.params['latent_dim']
num_layers_generator = trial.params['num_layers_generator']
neurons_per_layer_generator = trial.params['neurons_per_layer_generator']
num_layers_discriminator = trial.params['num_layers_discriminator']
neurons_per_layer_discriminator = trial.params['neurons_per_layer_discriminator']
opt = trial.params['optimizer']
#batch_size = trial.params['batch_size']
n_critic = trial.params['n_critic']
lambda_gp = trial.params['lambda_gp']

trainer = Trainer(data, latent_dim)

trainer.build_discriminator(num_layers=num_layers_discriminator, num_neurons=neurons_per_layer_discriminator)
trainer.discriminator.compile(loss=wasserstein_loss, optimizer=opt)

trainer.build_generator(num_layers=num_layers_generator, num_neurons=neurons_per_layer_generator)

trainer.build_combined()
trainer.combined.compile(loss=wasserstein_loss, optimizer=opt)

trainer.train(epochs=50, batch_size=32, n_critic=n_critic, lambda_gp=lambda_gp, save_interval=100)

trainer.save_model(save_dir_path)

print(f"Model data has been saved to {save_dir_path}")
