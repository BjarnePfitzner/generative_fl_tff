import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_grid_image(images, image_size, file_name):
    grid_dim = np.math.ceil(np.sqrt(images.shape[0]))
    fig = plt.figure(figsize=(image_size / 100 * grid_dim, image_size / 100 * grid_dim), dpi=300)

    for i in range(images.shape[0]):
        plt.subplot(grid_dim, grid_dim, i + 1)
        if images.shape[3] == 1:
            plt.imshow(images[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(images[i])
        plt.axis('off')

    plt.savefig(file_name, format='png')
    plt.close(fig)


def plot_2d_latent_space(model, label, data_dim, file_name, n=20):
    """Plots n x n digit images decoded from the 2d latent space."""
    scale = 1.0
    figsize = 15
    image = np.zeros((data_dim * n, data_dim * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = model([z_sample, np.array([label])])
            # Changed s.t. model is decoder only
            # x_decoded = tf.sigmoid(model((z_sample, np.array([label]))))
            digit = tf.reshape(x_decoded, (data_dim, data_dim))
            image[
            i * data_dim: (i + 1) * data_dim,
            j * data_dim: (j + 1) * data_dim,
            ] = digit

    fig = plt.figure(figsize=(figsize, figsize))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')

    plt.savefig(file_name, format='png')
    plt.close(fig)
