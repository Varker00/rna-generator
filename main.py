import json
import os
import time

import pandas as pd
from keras.optimizers import RMSprop

from GAN import GAN
from data_funcs import wasserstein_loss

data_path = 'original_data/GSE158508_normalized_counts.tsv'
data = pd.read_csv(data_path, sep='\t', index_col=0)

# set params here
num_layers_generator = 2
num_layers_discriminator = 2
neurons_per_layer_generator = 64
neurons_per_layer_discriminator = 64
learning_rate = 1e-6
n_critic = 1
lambda_gp = 1e-2
latent_dim = 64
batch_size = 32
opt = RMSprop(learning_rate=learning_rate, name="RMSprop")

# prepare to dump params to json
params = {
    "num_layers_generator": num_layers_generator,
    "num_layers_discriminator": num_layers_discriminator,
    "neurons_per_layer_generator": neurons_per_layer_generator,
    "neurons_per_layer_discriminator": neurons_per_layer_discriminator,
    "learning_rate": learning_rate,
    "n_critic": n_critic,
    "lambda_gp": lambda_gp,
    "latent_dim": latent_dim,
    "batch_size": batch_size,
    "opt": opt.get_config()['name']
}

current_time = time.time()
current_struct_time = time.localtime(current_time)
save_dir_path = os.path.join('generation_results', time.strftime("%Y-%m-%d--%H-%M", current_struct_time))

if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

model_params_file = os.path.join(save_dir_path, 'model_params.json')

with open(model_params_file, 'w') as json_file:
    json.dump(params, json_file, indent=4)
    print(f"JSON data has been saved to {model_params_file}")

gan = GAN(data, latent_dim=latent_dim)
gan.build_generator(num_layers_generator, neurons_per_layer_generator)
gan.build_discriminator(num_layers_discriminator, neurons_per_layer_discriminator)
gan.build_combined()

gan.compile(loss=wasserstein_loss, opt=opt)

gan.train(epochs=1, batch_size=batch_size, n_critic=n_critic, lambda_gp=lambda_gp)

gan.save_model(save_dir_path)
