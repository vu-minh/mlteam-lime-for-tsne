# minhvu
# 13/11/2019
# functions for generate samples

import math
import numpy as np


def sample_around(x, sigma=0.1, n_samples=1):
    """Sample a new point around the given datapoint `x`
    `x_sampled = x + epsilon`
    epsilon is a isotropic gaussian noise with variance `sigma`
    """
    epsilon = (sigma ** 2) * np.random.randn(n_samples, len(x.shape))
    return x + epsilon


def sample_with_global_noise(x, data, factor=0.5, n_samples=1):
    """Sample with global noise from the multivariate gaussian of HD `data` (denoted as `X`).
    global noise `epsilon` ~ MVN(X.mu, X.sigma^2)
    Given the input point `x`, return `x + epsilon`
    """
    mean = data.mean(axis=0)  # global mean of data according each feature
    sigma = data.var(axis=0)  # sigma is variance, which is square of std
    epsilon = np.random.multivariate_normal(mean=mean, cov=np.diag(sigma), size=n_samples)
    return x + factor * epsilon


def perturb_image(x, replace_rate=(1, 4)):
    """Random remove some pixels in the input image `x`
    The percent of removed pixels is in `replace_rate`.
    """
    replace_rate = 0.1 * np.random.randint(*replace_rate)
    mask = np.random.choice([0, 1], size=x.shape, p=((1 - replace_rate), replace_rate))
    x_new = x.copy()
    x_new[mask.astype(np.bool)] = 0
    return x_new.reshape(1, -1)


def remove_blob(x, n_remove=1):
    """Randomly remove some blobs with random size from the input image `x`
    """
    img_size = int(math.sqrt(len(x)))
    x_new = x.copy().reshape(img_size, img_size)
    for _ in range(n_remove):
        col1, col2 = np.random.randint(0, img_size, size=2)
        row1, row2 = np.random.randint(0, img_size, size=2)
        x_new[min(row1, row2) : max(row1, row2), min(col1, col2) : max(col1, col2)] = 0
    return x_new.reshape(1, -1)
