# minhvu
# 08/11/2019
# Query-blackbox like function for tsne
# Goal: given an input X, approximate an embedding y without fully rerun tsne

import os
import math
import joblib
from time import time
from functools import partial

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import utils
from sampling import sample_around, sample_with_global_noise
from sampling import perturb_image, remove_blob


def tsne_sample_embedded_points(
    X,
    selected_idx,
    n_samples=10,
    sigma_HD=0.1,
    sigma_LD=0.1,
    tsne_hyper_params={},
    early_stop_hyper_params={},
    sampling_method="sample_around",
    log_dir="",
    force_recompute=False,
    batch_mode=False,
):
    """Estimate sampled embedding in LD
    Args:
        X: input data [N, D]
        n_samples: number of points to sample around a selected point
        sigma_HD: variance for noise in HD
        sigma_LD: variance for noise in LD
        tsne_hyper_params: dict of tsne hyper-params.
        early_stop_hyper_params: dict of hyper-params for quick re-run t-SNE with early-stop
    Returns:
        Y: original embedding (Nx2)
        sample_ids: list of indices of sampled points
        s_samples: list of `n_samples` points HD
        y_samples: list of `n_samples` embedded points in LD
    """
    assert isinstance(selected_idx, int), "`selected_idx` must be a valid index"

    # first, run tsne with base `perplexity` to obtain the "base model"
    log_name_pattern = (
        f"-id{selected_idx}"
        f"-perp{tsne_hyper_params['perplexity']}"
        f"-sigmaHD{sigma_HD}-sigmaLD{sigma_LD}"
        f"-seed{tsne_hyper_params['random_state']}.Z"
    )
    log_name_Y = f"{log_dir}/Y{log_name_pattern}"
    if os.path.exists(log_name_Y) and not force_recompute:
        Y = joblib.load(log_name_Y)
        print("[DEBUG] Reuse: ", log_name_Y)
    else:
        tsne = TSNE(init="pca", **tsne_hyper_params)  # pca init for more stable
        Y = tsne.fit_transform(X)
        joblib.dump(Y, log_name_Y)
        print("[DEBUG] Initial tsne model: ", tsne)

    log_name_samples = f"{log_dir}/{sampling_method}{n_samples}{log_name_pattern}"
    if os.path.exists(log_name_samples) and not force_recompute:
        x_samples, y_samples = joblib.load(log_name_samples)
        print("[DEBUG] Reuse: ", log_name_samples)
    else:
        # prepare sampling function according to the chosen sampling method
        sampling_func = {
            "sample_around": partial(sample_around, sigma=sigma_HD),
            "sample_with_global_noise": partial(sample_with_global_noise, data=X),
            "perturb_image": partial(perturb_image, replace_rate=(1, 9)),
            "remove_blob": partial(remove_blob, n_remove=2),
        }[sampling_method]

        # generate samples in HD
        x_samples = [sampling_func(X[selected_idx]) for _ in range(n_samples)]
        x_samples = np.array(x_samples).reshape(-1, X.shape[1])

        # update hyper-params for quick re-run tsne
        tsne_hyper_params.update(early_stop_hyper_params)

        if batch_mode:
            tick = time()
            y_samples = query_blackbox_tsne(
                X=X,
                Y=Y,
                query_idx=selected_idx,
                query_points=x_samples,
                sigma_LD=sigma_LD,
                tsne_hyper_params=tsne_hyper_params,
            )
            print(
                f"[DEBUG] Query-blackbox for {n_samples} points in {time() - tick:.3f} seconds"
            )
        else:
            y_samples = []
            for i, x_sample in enumerate(x_samples):
                # ask tsne for the corresponding embedding of input sample
                tick = time()
                y_sample = query_blackbox_tsne(
                    X=X,
                    Y=Y,
                    query_idx=selected_idx,
                    query_points=x_sample,
                    sigma_LD=sigma_LD,
                    tsne_hyper_params=tsne_hyper_params,
                )
                print(
                    f"[DEBUG] Query-blackbox for {i+1}th point in {time() - tick:.3f} seconds"
                )
                y_samples.append(y_sample)
            y_samples = np.array(y_samples).reshape(-1, Y.shape[1])

        # save the sampled points for latter use
        joblib.dump((x_samples, y_samples), log_name_samples)
    return Y, x_samples, y_samples


def query_blackbox_tsne(
    X, Y, query_idx=None, query_points=None, sigma_LD=0.5, tsne_hyper_params={}
):
    """Query-blackbox-like function for sample tsne embedding.
    Given the original `tsne` model (represented by (X, Y) pair),
    predict the embedding for the new `query_points`.
    `query_points` is an array of input query points in HD, of shape [, D].
    """
    assert query_idx is not None and query_points is not None
    N = X.shape[0]

    # sample proposal embedded position for the input `query_idx`
    # y_proposal = np.zeros((1, 2))
    y_proposals = sample_around(Y[query_idx], sigma=sigma_LD, n_samples=len(query_points))

    # append new samples in HD and LD into X and Y
    X_new = np.concatenate([X, query_points], axis=0)
    Y_new = np.concatenate([Y, y_proposals], axis=0)

    # approximate the original tsne model by a new one with initialized embedding
    # given by the original model
    tsne = TSNE(init=Y_new, **tsne_hyper_params)
    Y_with_samples = tsne._fit(X_new, skip_num_points=N)

    # debug to see if the original embedding does not change (or change a little bit)
    # TODO: discuss if it is necessary to make the initial Y does not change
    if tsne_hyper_params.get("verbose", 0) > 0:
        diff = Y - Y_with_samples[:N]
        print("[DEBUG] Check if the embedding for N original points does not change")
        print("Norm of diff: ", np.linalg.norm(diff))
        print("Sum square error: ", 0.5 * np.sum(np.dot(diff.T, diff)))

    return Y_with_samples[N:]


if __name__ == "__main__":
    # Demo how to run the sampling task
    # For a full workflow, see `simpler_explainer.py`

    n_samples = 10  # number of points to sample
    sigma_HD = 1.0  # larger of Gaussian in HD
    sigma_LD = 1.0  # larger of Gaussian in LD, has less effect when the re-run of tsne stable
    N_max = 200  # maximum number of data points for testing only
    sampling_method = "sample_around"  # in ["sample_around", "sample_with_global_noise"]
    dataset_name = "iris"
    plot_dir = f"./plots/{dataset_name}"
    log_dir = f"./var/{dataset_name}"
    data_home = "./data"
    debug_level = 0

    # TODO check stability of sampling. (e.g. seed 1024 iris, 42 digits)

    # basic params to run tsne the first time
    tsne_hyper_params = dict(perplexity=30, n_iter=1500, random_state=42, verbose=debug_level)

    # to re-run tsne quickly, take the initial embedding as `init` and use the following params
    early_stop_hyper_params = dict(
        early_exaggeration=1,  # disable exaggeration
        n_iter_without_progress=50,
        min_grad_norm=1e-7,
        n_iter=500,  # many iterations make the embedding of the sampled points more stable
        verbose=debug_level,
    )

    X, labels = {"iris": load_iris, "digits": load_digits}[dataset_name](return_X_y=True)
    X, labels = X[:N_max], labels[:N_max]
    X = StandardScaler().fit_transform(X)

    # set numpy random seed for reproducing
    np.random.seed(seed=tsne_hyper_params.get("random_state", 42))

    # select a point to sample in HD
    selected_idx = np.random.randint(X.shape[0])

    # create new samples in HD and embed them in LD
    Y, x_samples, y_samples = tsne_sample_embedded_points(
        X,
        selected_idx=selected_idx,
        n_samples=n_samples,
        sigma_HD=sigma_HD,
        sigma_LD=sigma_LD,
        tsne_hyper_params=tsne_hyper_params,
        early_stop_hyper_params=early_stop_hyper_params,
        log_dir=log_dir,
        force_recompute=False,
    )

    # viz the original embedding with the new sampled points
    out_name = f"{plot_dir}/{sigma_HD}-{sigma_LD}-{n_samples}_scatter.png"
    utils.scatter_with_samples(Y, y_samples, selected_idx, labels=labels, out_name=out_name)
