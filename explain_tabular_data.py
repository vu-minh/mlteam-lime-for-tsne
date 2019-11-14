# minh and geraldin
# 14/11/2019
# demo explaining tabular data

import os
import joblib

import numpy as np
from matplotlib import pyplot as plt

from sample_tsne import tsne_sample_embedded_points
from simple_explainer import explain_samples
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression

from utils import scatter_with_samples_with_text


def _load_country():
    data = joblib.load("./dataset/country.dat")
    return data["X"], data["country_names"], data["indicator_names"]


def _load_wine():
    data = load_wine()
    return data["data"], data["target"], data["feature_names"]


def _load_iris():
    data = load_iris()
    return data["data"], data["target"], data["feature_names"]


def load_tabular_dataset(dataset_name="country", standardize=True):
    """Load the tabular dataset with the given `dataset_name`
    Returns:
        X: [N x D] data itself
        labels: [N]
        feature_names: [D]
    """
    load_func = {"country": _load_country, "wine": _load_wine, "iris": _load_iris}[dataset_name]

    X, labels, feature_names = load_func()
    if standardize:
        X = StandardScaler().fit_transform(X)
        # X = Normalizer().fit_transform(X)
        # X -= X.mean(axis=0, keepdims=True)
    return X, labels, feature_names


def plot_weights(W, feature_names, out_name="noname00"):
    assert W is not None, "Error with linear model!"
    assert W.shape[1] == len(feature_names)
    n_cols = W.shape[0]
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 6, 6), sharey=True)
    for ax, weights in zip(axes.ravel(), W):

        print(weights)
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, weights)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        # ax.invert_yaxis()  # labels read top-to-bottom
        # ax.set_xlabel("Importance of features")

    fig.savefig(out_name)
    plt.close(fig)


if __name__ == "__main__":
    # define the variables for the dataset name, number of samples, ...
    n_samples = 50  # number of points to sample
    sigma_HD = 0.25  # larger of Gaussian in HD
    sigma_LD = 0.5  # larger of Gaussian in LD
    seed = 42  # for reproducing
    debug_level = 0  # verbose in tsne
    N_max = 500  # maximum number of data points for testing only
    force_recompute = True  # use pre-calculated embedding and samples or recompute them
    sampling_method = "sample_around"  # add noise to selected point, works with tabular data

    dataset_name = "iris"
    log_dir = f"./var/{dataset_name}"
    plot_dir = f"./plots/{dataset_name}"
    for a_dir in [plot_dir, log_dir]:
        if not os.path.exists(a_dir):
            os.mkdir(a_dir)

    # set numpy random seed for reproducing
    np.random.seed(seed)

    # basic params to run tsne the first time
    tsne_hyper_params = dict(perplexity=30, n_iter=1500, random_state=seed, verbose=debug_level)

    # to re-run tsne quickly, take the initial embedding as `init` and use the following params
    early_stop_hyper_params = dict(
        early_exaggeration=1,  # disable exaggeration
        n_iter_without_progress=100,
        min_grad_norm=1e-7,
        n_iter=1500,  # increase it to test stability
        verbose=debug_level,
    )

    # load the chosen dataset
    X, labels, feature_names = load_tabular_dataset(dataset_name, standardize=True)

    # select a (random) point to explain
    selected_idx = np.random.randint(X.shape[0])
    print("[DEBUG] selected point: ", selected_idx, labels[selected_idx])

    # apply the workflow for generating the samples in HD and embed them in LD
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
        batch_mode=False,
    )

    # viz the original embedding with the new sampled points
    out_name_prefix = f"{plot_dir}/{sigma_HD}-{sigma_LD}-{n_samples}"
    out_name_Y = f"{out_name_prefix}_scatter.png"
    # TODO show the country with name or numeric `labels`
    scatter_with_samples_with_text(
        Y, y_samples, selected_idx, texts=labels, out_name=out_name_Y
    )

    # apply the linear model for explaining the sampled points
    W = explain_samples(
        x_samples,
        y_samples,
        linear_model=LinearRegression(),  # ElasticNet(alpha=1.0, l1_ratio=0.1),
        find_rotation=True,
    )

    # visualize the weights of the linear model
    # (show contribution of the most important features)
    out_name_W = f"{out_name_prefix}_explanation.png"
    plot_weights(W, feature_names, out_name=out_name_W)
