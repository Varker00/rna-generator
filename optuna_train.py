import os
import time
import logging
import tensorflow as tf
import optuna
from data_process import DataProcessor
from GAN_models import Generator, Discriminator
from GAN_trainer import GANTrainer
from WGAN_trainer import WGANTrainer
from WGANWC_trainer import WGANWCTrainer
from WGANGP_trainer import WGANGPTrainer
from utils import save_results, plot_results, save_hyperparams, calculate_fd, inverse_scale_data
from global_variables import EPOCHS

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


MODEL = 'GAN'           # 'GAN', 'WGAN', 'WGANWC', 'WGAN-GP'
NUM_FEATURES = 200      # 25, 50, 100, 200, 500, 1000, 2000, 5000
# EPOCHS = 1000

def objective(trial):
    latent_dim = trial.suggest_int('latent_dim', 32, 128)
    num_layers_generator = trial.suggest_int('num_layers_generator', 2, 6)
    num_neurons_generator = trial.suggest_int('num_neurons_generator', 32, 512)
    num_layers_discriminator = trial.suggest_int('num_layers_discriminator', 2, 6)
    num_neurons_discriminator = trial.suggest_int('num_neurons_discriminator', 32, 512)
    dropout_discriminator = trial.suggest_float('dropout_discriminator', 0.0, 0.4)
    batch_size = 2**trial.suggest_int('batch_size', 0, 6)
    g_lr = trial.suggest_float('g_lr', 1e-7, 1e-3)
    d_lr = trial.suggest_float('d_lr', 1e-7, 1e-3)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    if MODEL == 'WGAN' or MODEL == 'WGANWC' or MODEL == 'WGANGP':
        n_critic = trial.suggest_int('n_critic', 1, 10)
    else:
        n_critic = -1
    if MODEL == 'WGANWC':
        weight_clip = trial.suggest_float('weight_clip', 0.001, 0.1)
    else:
        weight_clip = -1
    if MODEL == 'WGANGP':
        lambda_gp = trial.suggest_float('lambda_gp', 0.1, 10.0)
    else:
        lambda_gp = -1


    # Identification of the experiment
    id = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())

    data_path = 'original_data/GSE158508_normalized_counts.tsv'
    synthetic_data_path = 'synthetic_data/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id)
    checkpoint_dir = 'checkpoints/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id)
    results_path = 'results/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id)
    models_path = 'models/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id)

    if not os.path.exists(synthetic_data_path):
        os.makedirs(synthetic_data_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists('results/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id)):
        os.makedirs('results/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id))
    if not os.path.exists('synthetic_data/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id)):
        os.makedirs('synthetic_data/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id))
    if not os.path.exists('models/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id)):
        os.makedirs('models/{}/{}/{}/{}/'.format(MODEL, NUM_FEATURES, EPOCHS, id))


    hyperparams = {
        'latent_dim': latent_dim,
        'num_layers_generator': num_layers_generator,
        'num_neurons_generator': num_neurons_generator,
        'num_layers_discriminator': num_layers_discriminator,
        'num_neurons_discriminator': num_neurons_discriminator,
        'dropout_discriminator': dropout_discriminator,
        'batch_size': batch_size,
        'g_lr': g_lr,
        'd_lr': d_lr,
        'optimizer': optimizer,             # 'Adam' or 'RMSprop'
        'n_critic': n_critic,               # used in WGAN, WGAN-WC, WGAN-GP
        'weight_clip': weight_clip,         # used in WGAN-WC
        'lambda_gp': lambda_gp              # used in WGAN-GP
    }

    # Model's hyperparameters
    latent_dim = hyperparams['latent_dim']
    num_layers_generator = hyperparams['num_layers_generator']
    num_neurons_generator = hyperparams['num_neurons_generator']
    num_layers_discriminator = hyperparams['num_layers_discriminator']
    num_neurons_discriminator = hyperparams['num_neurons_discriminator']
    dropout_discriminator = hyperparams['dropout_discriminator']
    
    # Training hyperparameters
    epochs = EPOCHS
    save_interval = 50
    batch_size = hyperparams['batch_size']
    generator_lr = hyperparams['g_lr']
    discriminator_lr = hyperparams['d_lr']
    optimizer = hyperparams['optimizer']
    n_critic = hyperparams['n_critic']
    weight_clip = hyperparams['weight_clip']
    lambda_gp = hyperparams['lambda_gp']

    # Saving hyperparameters
    save_hyperparams(hyperparams, '{}/hyperparams.json'.format(results_path))

    # Loading and preprocessing data
    processor = DataProcessor(data_path)
    processor.load_and_preprocess(NUM_FEATURES)
    processed_data = processor.get_data()
    
    # Creating models
    generator = Generator(latent_dim, processed_data.shape[1], num_layers_generator, num_neurons_generator).get_model()
    discriminator = Discriminator(processed_data.shape[1], num_layers_discriminator, num_neurons_discriminator, dropout_discriminator).get_model()

    # Creating trainer
    if MODEL == 'GAN':
        gan_trainer = GANTrainer(processor, generator, discriminator, latent_dim, generator_lr, discriminator_lr, optimizer, id)
    elif MODEL == 'WGAN':
        gan_trainer = WGANTrainer(processor, generator, discriminator, latent_dim, generator_lr, discriminator_lr, optimizer, id, n_critic)
    elif MODEL == 'WGANWC':
        gan_trainer = WGANWCTrainer(processor, generator, discriminator, latent_dim, generator_lr, discriminator_lr, optimizer, id, n_critic, weight_clip)
    elif MODEL == 'WGANGP':
        gan_trainer = WGANGPTrainer(processor, generator, discriminator, latent_dim, generator_lr, discriminator_lr, optimizer, id, n_critic, lambda_gp)
    
    # Training GAN
    d_loss, g_loss, gradient_norm, fd_scores = gan_trainer.train(processed_data, epochs, batch_size, save_interval, checkpoint_dir)

    # Saving and plotting results
    save_results(d_loss, g_loss, gradient_norm, fd_scores, results_path)
    plot_results(d_loss, g_loss, gradient_norm, fd_scores, save_interval, results_path)

    # Generating data
    num_samples_to_generate = processed_data.shape[0]
    generated_data = gan_trainer.generate_data(num_samples_to_generate)

    # Saving generated data
    processor.load_and_postprocess(generated_data)
    processor.save_generated_data('{}/generated_data.tsv'.format(synthetic_data_path))

    # Saving models
    gan_trainer.save_models(models_path)

    original_data = processor.get_original_data()
    original_min, original_max = original_data.min().min(), original_data.max().max()
    generated_data = gan_trainer.generate_data(original_data.shape[0])
    generated_data = inverse_scale_data(generated_data, original_min, original_max)
    fd_score = calculate_fd(original_data, generated_data)

    return fd_score


def main():
    sqlite_url = "sqlite:///baza.db"

    study_name = f"{MODEL}_{NUM_FEATURES}_{EPOCHS}_study"

    study = optuna.create_study(study_name=study_name, storage=sqlite_url, load_if_exists=True, direction='minimize')

    study.optimize(objective, n_trials=3)

    print('Best trial for', study_name) 
    trial = study.best_trial
    print(' Value: ', trial.value)
    print(' Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

if __name__ == '__main__':
    main()