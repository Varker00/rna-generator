import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from sklearn.utils.extmath import randomized_svd


def gradient_penalty(real_data, fake_data, discriminator):
    alpha = tf.random.normal([real_data.shape[0], 1], 0.0, 1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        prediction = discriminator(interpolated)
    gradients = tape.gradient(prediction, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def compute_gradient_norm(model, real_data, fake_data):
    with tf.GradientTape() as tape:
        # Obliczanie straty dyskryminatora
        real_output = model(real_data, training=True)
        fake_output = model(fake_data, training=True)
        loss = wasserstein_loss(real_output, fake_output)

    # Obliczanie gradientów względem wag dyskryminatora
    gradients = tape.gradient(loss, model.trainable_variables)
    # Obliczanie normy gradientów
    gradient_norm = tf.linalg.global_norm(gradients)
    return gradient_norm.numpy()


def matrix_sqrt_via_svd(matrix):
    """ Przybliżony pierwiastek macierzy przy użyciu randomized SVD. """
    U, S, V = randomized_svd(matrix, n_components=100)
    sqrt_S = np.sqrt(S)
    return np.dot(U * sqrt_S, V)

def scale_data(data):
    data = (data - data.min().min()) / (data.max().max() - data.min().min())
    data = data * 2 - 1
    return data

def inverse_scale_data(generated_data, original_min, original_max):
    generated_data = (generated_data + 1) / 2
    generated_data = generated_data * (original_max - original_min) + original_min
    return generated_data


def calculate_fid(real_data, generated_data):
    # Przygotowanie danych
    real_data = real_data.astype('float32')
    generated_data = generated_data.astype('float32')

    # Obliczanie średniej i macierzy kowariancji
    mu1, sigma1 = real_data.mean(axis=0), np.cov(real_data, rowvar=False)
    mu2, sigma2 = generated_data.mean(axis=0), np.cov(generated_data, rowvar=False)

    # Dodanie stabilizacji numerycznej do macierzy kowariancji
    sigma1 += np.eye(sigma1.shape[0]) * 1e-6
    sigma2 += np.eye(sigma2.shape[0]) * 1e-6

    ssdiff = tf.reduce_sum(tf.square(mu1 - mu2), axis=-1)

    # Obliczanie przybliżonego pierwiastka z iloczynu macierzy kowariancji
    sqrt_prod = matrix_sqrt_via_svd(sigma1 @ sigma2)

    # Obliczanie śladu (trace) macierzy kowariancji i pierwiastka z ich iloczynu
    trace_sum = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_prod)

    ssdiff = tf.cast(ssdiff, tf.float32)
    trace_sum = tf.cast(trace_sum, tf.float32)
    print(f"ssdiff: {ssdiff}, trace_sum: {trace_sum}")

    fid_score = ssdiff + trace_sum

    return fid_score

def save_results(d_loss_list, g_loss_list, gradient_norm_list, fid_score_list, file_path):
    fid_scores = [score.numpy() if isinstance(score, tf.Tensor) else score for score in fid_score_list]
    results = pd.DataFrame({
        'd_loss': d_loss_list,
        'g_loss': g_loss_list,
        'gradient_norm': gradient_norm_list,
    })
    fid_results = pd.DataFrame({
        'fid_score': fid_scores
    })
    file_path = file_path + 'results_1.csv'
    for i in range(2, 10):
        if not os.path.exists(file_path):
            break
        file_path = file_path[:-5] + str(i) + '.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    results.to_csv(file_path, index=False)
    file_path = file_path[:-4] + '_fid.csv'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fid_results.to_csv(file_path, index=False)

def plot_results(d_loss_list, g_loss_list, gradient_norm_list, fid_score_list, file_path):
    plt.figure(figsize=(10, 12))

    # Plotting discriminator and generator losses
    plt.subplot(3, 1, 1)
    plt.plot(d_loss_list, label='Discriminator Loss')
    plt.plot(g_loss_list, label='Generator Loss')
    plt.title('GAN Losses')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting gradient norm
    plt.subplot(3, 1, 2)
    plt.plot(gradient_norm_list, label='Gradient Norm', color='orange')
    plt.title('Gradient Norm')
    plt.ylabel('Value')
    plt.legend()

    # Plotting FD scores
    plt.subplot(3, 1, 3)
    plt.plot(fid_score_list, label='FD Score', color='green')
    plt.title('FD Score')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    file_path = file_path + 'results_1.png'
    for i in range(2, 10):
        if not os.path.exists(file_path):
            break
        file_path = file_path[:-6] + str(i) + '.png'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
   # plt.show(block=False)

def save_hyperparams(hyperparams, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(hyperparams, file, indent=4)

def remove_checkpoints(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".index") or file.endswith(".data-00000-of-00001"):
            os.remove(os.path.join(checkpoint_dir, file))