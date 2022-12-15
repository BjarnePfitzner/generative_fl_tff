from functools import partial

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm


def scale_element(elem, new_shape):
    if elem.shape[-1] == 1:
        elem = tf.image.grayscale_to_rgb(elem)
    return tf.image.resize(elem, new_shape, method='bilinear', antialias=True)


# calculate frechet inception distance
def calculate_fid(inception_model, generated_images, real_activations):
    generated_images = generated_images.map(partial(scale_element, new_shape=[299, 299]),
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Run fake data generator for same number of steps as real batches
    fake_activations = inception_model.predict(generated_images)
    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
    mu2, sigma2 = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
