# minhvu
# 12/11/2019
# test explain the sampled points by t-SNE

import os
import math

import numpy as np
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, ElasticNet

from common.dataset import dataset
from sample_tsne import tsne_sample_embedded_points
from sampling import perturb_image, remove_blob
from utils import scatter_with_samples, plot_samples, plot_heatmap


def explain_samples(x_samples, y_samples):
    """Naive explainer by a linear regression (LR) model.
    Note that, we do not use the intercept, so we should standardize the input.
    Args:
        x_samples: [n_samples x D]
        y_samples: [n_samples x 2]
    Returns:
        weights of LR model (representing the importance of each pixel)
    """
    # x_samples = StandardScaler().fit_transform(x_samples)
    # x_samples = x_samples - x_samples.mean(axis=0, keepdims=True)

    reg = ElasticNet(alpha=0.5, l1_ratio=0.75).fit(x_samples, y_samples)
    return reg.coef_


def test_remove_blob(x, n_repeat=16, n_remove=1):
    x_samples = []
    for i in range(n_repeat):
        x_new = remove_blob(x)
        x_samples.append(x_new)
    plot_samples(np.array(x_samples), out_name=f"{plot_dir}/test_remove_blob{n_remove}.png")


def run_explainer():
    """Test workflow: generate samples, apply a linear model and try to explain the weights
    Note: using global variables in __main__ (not good practice)
    """

    # TODO check stability of sampling. (e.g. seed 1024 iris, 42 digits)
    # set numpy random seed for reproducing
    np.random.seed(seed)

    # basic params to run tsne the first time
    tsne_hyper_params = dict(perplexity=50, n_iter=1500, random_state=seed, verbose=debug_level)

    # to re-run tsne quickly, take the initial embedding as `init` and use the following params
    early_stop_hyper_params = dict(
        early_exaggeration=1,  # disable exaggeration
        n_iter_without_progress=50,
        min_grad_norm=1e-7,
        n_iter=1000,  # increase it to test stability
        verbose=debug_level,
    )

    dataset.set_data_home(data_home)
    X_original, X, labels = dataset.load_dataset(dataset_name)
    X_original, X, labels = shuffle(X_original, X, labels, n_samples=N_max, random_state=seed)

    # select a point to sample in HD
    selected_idx = np.random.randint(X.shape[0])
    print("[DEBUG] selected point: ", selected_idx, labels[selected_idx])

    # create new samples in HD and embed them in LD
    Y, x_samples, y_samples = tsne_sample_embedded_points(
        X,
        selected_idx=selected_idx,
        n_samples=n_samples,
        sigma_HD=sigma_HD,
        sigma_LD=sigma_LD,
        sampling_method=sampling_method,
        tsne_hyper_params=tsne_hyper_params,
        early_stop_hyper_params=early_stop_hyper_params,
        log_dir=log_dir,
        force_recompute=force_recompute,
    )

    # viz the original embedding with the new sampled points
    out_name_prefix = f"{plot_dir}/{sigma_HD}-{sigma_LD}-{n_samples}"
    out_name = f"{out_name_prefix}_scatter.png"
    scatter_with_samples(Y, y_samples, selected_idx, labels=labels, out_name=out_name)

    # plot the weights of the linear model
    W = explain_samples(x_samples, y_samples)
    plot_heatmap(W, img=X_original[selected_idx], out_name=f"{out_name_prefix}_explanation.png")

    # show the sampled images in HD
    plot_samples(x_samples, out_name=f"{out_name_prefix}_samples.png")


if __name__ == "__main__":
    # TODO: turn these variables to input arguments with argparse
    n_samples = 10  # number of points to sample
    sigma_HD = 0.5  # larger of Gaussian in HD
    sigma_LD = 0.5  # larger of Gaussian in LD
    seed = 8888  # for reproducing
    debug_level = 0  # verbose in tsne
    N_max = 500  # maximum number of data points for testing only
    force_recompute = False  # use pre-calculated embedding and samples or recompute them

    # sampling method in ["sample_around", "sample_with_global_noise", "perturb_image", "remove_blob"]
    sampling_method = "remove_blob"
    data_home = "./data"  # local data on my computer
    dataset_name = "MNIST"
    plot_dir = f"./plots/{dataset_name}"
    log_dir = f"./var/{dataset_name}"
    for a_dir in [plot_dir, log_dir]:
        if not os.path.exists(a_dir):
            os.mkdir(a_dir)

    run_explainer()

    # test_remove_blob(X[89], n_remove=1)
    # test_remove_blob(X[0], n_remove=2)
    # test_remove_blob(X[123], n_remove=3)
