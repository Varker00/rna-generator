import os
import time
import logging
import tensorflow as tf
from data_process import DataProcessor
from GAN_models import Generator, Discriminator
from GAN_trainer import GANTrainer
from utils import save_results, plot_results, save_hyperparams

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def main():
    MODEL = 'GAN'
    NUM_FEATURES = 200

    # Identyfikator modelu
    id = time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())

    data_path = 'original_data/GSE158508_normalized_counts.tsv'
    synthetic_data_path = 'synthetic_data/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id)
    checkpoint_dir = 'checkpoints/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id)
    results_path = 'results/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id)
    models_path = 'models/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id)
    
    if not os.path.exists(synthetic_data_path):
        os.makedirs(synthetic_data_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists('results/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id)):
        os.makedirs('results/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id))
    if not os.path.exists('synthetic_data/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id)):
        os.makedirs('synthetic_data/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id))
    if not os.path.exists('models/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id)):
        os.makedirs('models/{}/{}/{}/'.format(MODEL, NUM_FEATURES, id))

    hyperparams = {
        'latent_dim': 64,
        'num_layers_generator': 4,
        'num_neurons_generator': 256,
        'num_layers_discriminator': 2,
        'num_neurons_discriminator': 128,
        'dropout_discriminator': 0.2,
        'batch_size': 16,
        'g_lr': 0.0001,
        'd_lr': 0.0001,
        'optimizer': 'RMSprop'
    }

    # Parametry GAN
    latent_dim = hyperparams['latent_dim']
    num_layers_generator = hyperparams['num_layers_generator']
    num_neurons_generator = hyperparams['num_neurons_generator']
    num_layers_discriminator = hyperparams['num_layers_discriminator']
    num_neurons_discriminator = hyperparams['num_neurons_discriminator']
    dropout_discriminator = hyperparams['dropout_discriminator']
    
    # Parametry trenowania
    epochs = 5000
    batch_size = hyperparams['batch_size']
    generator_lr = hyperparams['g_lr']
    discriminator_lr = hyperparams['d_lr']
    optimizer = hyperparams['optimizer']
    save_interval = 100

    # Zapisywanie hiperparametrów
    save_hyperparams(hyperparams, 'results/{}/{}/{}/hyperparams.json'.format(MODEL, NUM_FEATURES, id))

    # Wczytywanie i przetwarzanie danych
    processor = DataProcessor(data_path)
    processor.load_and_preprocess(NUM_FEATURES)
    processed_data = processor.get_data()
    
    # Tworzenie modeli GAN
    generator = Generator(latent_dim, processed_data.shape[1], num_layers_generator, num_neurons_generator).get_model()
    discriminator = Discriminator(processed_data.shape[1], num_layers_discriminator, num_neurons_discriminator, dropout_discriminator).get_model()

    # Tworzenie i kompilacja trenera GAN
    gan_trainer = GANTrainer(processor, generator, discriminator, latent_dim, generator_lr, discriminator_lr, optimizer, id)

    # Trenowanie GAN
    d_loss, g_loss, gradient_norm, fid_scores = gan_trainer.train(processed_data, epochs, batch_size, save_interval, checkpoint_dir)

    # Zapisywanie i wykresy wyników
    save_results(d_loss, g_loss, gradient_norm, fid_scores, results_path)
    plot_results(d_loss, g_loss, gradient_norm, fid_scores, results_path)

    # Dotrenowywanie GAN
    epochs = 5000
    d_loss, g_loss, gradient_norm, fid_scores = gan_trainer.train(processed_data, epochs, batch_size, save_interval, checkpoint_dir)

    # Zapisywanie i wykresy wyników
    save_results(d_loss, g_loss, gradient_norm, fid_scores, results_path)
    plot_results(d_loss, g_loss, gradient_norm, fid_scores, results_path)

    # Generowanie danych
    num_samples_to_generate = processed_data.shape[0]
    generated_data = gan_trainer.generate_data(num_samples_to_generate)

    # Zapisanie wygenerowanych danych do pliku
    processor.load_and_postprocess(generated_data)
    processor.save_generated_data('{}/generated_data.tsv'.format(synthetic_data_path))

    # Zapisywanie modeli
    gan_trainer.save_models(models_path)

if __name__ == '__main__':
    main()