# minhvu
# 13/11/2019
# functions for generate samples

import math
import numpy as np
from scipy.spatial.distance import cdist


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


def generate_samples_HD(
    selected_point, sampling_method="sample_around", sigma_HD=1.0, n_samples=100
):
    """Generate `n_samples` "around" the `selected_point`
    """
    sampling_func = {
        "sample_around": partial(sample_around, sigma=sigma_HD),
        "perturb_image": partial(perturb_image, replace_rate=(1, 9)),
        "remove_blob": partial(remove_blob, n_remove=2),
    }[sampling_method]
    return [sampling_func(selected_point) for _ in range(n_samples)]


def generate_samples_SMOTE(selected_id, X, k_nearbors=10, n_samples=100):
    """Mimic SMOTE to generate sample between `selected_id`
        and one of its `k_neighbors`
    """
    # distance from the selected points to all other points in HD
    dist = cdist(X[selected_id].reshape(1, -1), X)
    neighbor_ids = np.argsort(dist)[0, 1 : k_nearbors + 1]
    print("[DEBUG] List neighbors: ", neighbor_ids)

    n_samples_per_pair = n_samples // k_nearbors
    x_samples = []
    for neighbor_idx in neighbor_ids:
        diff = X[neighbor_idx] - X[selected_id]
        for _ in range(n_samples_per_pair):
            alpha = np.clip(np.random.rand(), a_min=1e-3, a_max=1.0)
            x_sample = X[selected_id] + alpha * diff
            x_samples.append(x_sample.reshape(1, -1))

    print("[DEBUG] Generated samples: ", len(x_samples))
    return x_samples
