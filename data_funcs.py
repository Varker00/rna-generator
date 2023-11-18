import time
import numpy as np
import tensorflow as tf
from sklearn.utils.extmath import randomized_svd


def scale_data(data):
    min_val = data.min().min()  # Get the global minimum
    max_val = data.max().max()  # Get the global maximum
    scaled_data = (data - min_val) / (max_val - min_val)  # Apply min-max scaling
    return scaled_data


def inverse_scale_data(generated_data, original_min, original_max):
    generated_data = (generated_data + 1) / 2
    generated_data = generated_data * (original_max - original_min) + original_min
    return generated_data


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


def matrix_sqrt_newton_schulz(A, num_iters=10):
    dim = tf.shape(A)[0]
    Y = A
    I = tf.eye(dim, dtype=A.dtype)
    Z = tf.eye(dim, dtype=A.dtype)

    for i in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y


def matrix_sqrt_via_svd(matrix):
    """ Przybliżony pierwiastek macierzy przy użyciu randomized SVD. """
    U, S, V = randomized_svd(matrix, n_components=100)
    sqrt_S = np.sqrt(S)
    return np.dot(U * sqrt_S, V)


def calculate_fid(real_data, generated_data):
    # Przygotowanie danych
    start = time.time()
    real_data = real_data.astype('float32')
    generated_data = generated_data.astype('float32')
    print("Prepared data in {} seconds".format(time.time() - start))

    # Obliczanie średniej i macierzy kowariancji
    start = time.time()
    mu1, sigma1 = real_data.mean(axis=0), np.cov(real_data, rowvar=False)
    print("Computed real data mean and covariance in {} seconds".format(time.time() - start))

    start = time.time()
    mu2, sigma2 = generated_data.mean(axis=0), np.cov(generated_data, rowvar=False)
    print("Computed generated data mean and covariance in {} seconds".format(time.time() - start))

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
