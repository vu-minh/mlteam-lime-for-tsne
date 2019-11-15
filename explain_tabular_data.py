# minh and geraldin
# 14/11/2019
# demo explaining tabular data

import os
import joblib

import numpy as np
from matplotlib import pyplot as plt

from sample_tsne import tsne_sample_embedded_points
from simple_explainer import explain_samples
from sklearn.datasets import load_wine, load_iris, load_boston
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression

from utils import scatter_with_samples, plot_weights


def load_country():
    data = joblib.load("./dataset/country.dat")
    return {
        "data": data["X"],
        "target_names": data["country_names"],
        "feature_names": data["indicator_descriptions"],
    }


def load_tabular_dataset(dataset_name="country", standardize=True):
    """Load the tabular dataset with the given `dataset_name`
    Returns:
        X: [N x D] data itself
        labels: [N]
        feature_names: [D]
    """
    load_func = {
        "country": load_country,
        "wine": load_wine,
        "iris": load_iris,
        "boston": load_boston,
    }[dataset_name]

    data = load_func()
    X, label_names, feature_names = (
        data["data"],
        data["target_names"],
        data["feature_names"],
    )
    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, label_names, feature_names


if __name__ == "__main__":
    # define the variables for the dataset name, number of samples, ...
    n_samples = 100  # number of points to sample
    sigma_HD = 1.0  # larger of Gaussian in HD
    sigma_LD = 1.0  # larger of Gaussian in LD
    seed = 1024  # for reproducing
    debug_level = 0  # verbose in tsne
    N_max = 1000  # maximum number of data points for testing only
    force_recompute = False  # use pre-calculated embedding and samples or recompute them
    sampling_method = "sample_around"  # add noise to selected point, works with tabular data

    dataset_name = "country"
    log_dir = f"./var/{dataset_name}"
    plot_dir = f"./plots/{dataset_name}"
    for a_dir in [plot_dir, log_dir]:
        if not os.path.exists(a_dir):
            os.mkdir(a_dir)

    # set numpy random seed for reproducing
    np.random.seed(seed)

    # basic params to run tsne the first time
    tsne_hyper_params = dict(
        method="exact", perplexity=10, n_iter=1000, random_state=seed, verbose=debug_level
    )

    # to re-run tsne quickly, take the initial embedding as `init` and use the following params
    early_stop_hyper_params = dict(
        early_exaggeration=1,  # disable exaggeration
        n_iter_without_progress=100,
        min_grad_norm=1e-7,
        n_iter=500,  # increase it to test stability
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
    scatter_with_samples(Y, y_samples, selected_idx, texts=labels, out_name=out_name_Y)

    # apply the linear model for explaining the sampled points
    W, score, rotation = explain_samples(
        x_samples,
        y_samples,
        linear_model=Lasso(fit_intercept=False, alpha=0.015),
        # LinearRegression(),
        # ElasticNet(alpha=2.0, l1_ratio=0.1),
        find_rotation=True,
    )

    # visualize the weights of the linear model
    # (show contribution of the most important features)
    out_name_W = f"{out_name_prefix}_explanation.png"
    title = f"Best score $R^2$ = {score:.3f}, best rotation = {rotation} deg"
    plot_weights(W, feature_names, title=title, out_name=out_name_W)
