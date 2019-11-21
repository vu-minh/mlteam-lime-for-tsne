# minhvu
# 08/11/2019
# Query-blackbox like function for tsne
# Goal: given an input X, approximate an embedding y without fully rerun tsne

import os
import math
import joblib
from copy import deepcopy
from time import time
from functools import partial

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import utils
from sampling import sample_around
from sampling import generate_samples_HD, generate_samples_SMOTE
from modified_tsne import modifiedTSNE


def tsne_sample_embedded_points(
    X,
    selected_idx,
    n_samples=100,
    n_neighbors_SMOTE=10,
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
        n_neighbors_SMOTE: number of neighbors in HD to perform oversampling w. SMOTE
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
    N, D = X.shape

    # first, run tsne with base `perplexity` to obtain the "base model"
    log_name_pattern = (
        f"-id{selected_idx}"
        f"-perp{tsne_hyper_params['perplexity']}"
        f"-seed{tsne_hyper_params['random_state']}.Z"
    )
    log_name_Y = f"{log_dir}/Y{log_name_pattern}"
    if os.path.exists(log_name_Y) and not force_recompute:
        Y = joblib.load(log_name_Y)
        print("[DEBUG] Reuse: ", log_name_Y)
    else:
        tsne = modifiedTSNE(init="pca", **tsne_hyper_params)  # pca init for more stable
        Y = tsne.fit_transform(X)
        joblib.dump(Y, log_name_Y)
        print("[DEBUG] Initial tsne model: ", tsne)

    log_name_samples = f"{log_dir}/{sampling_method}{n_samples}{log_name_pattern}"
    if os.path.exists(log_name_samples) and not force_recompute:
        x_samples, y_samples = joblib.load(log_name_samples)
        print("[DEBUG] Reuse: ", log_name_samples)
    else:

        # generate samples in HD
        # x_samples = generate_samples_HD(selected_point=X[selected_idx])
        x_samples = generate_samples_SMOTE(
            selected_idx, X, k_nearbors=n_neighbors_SMOTE, n_samples=n_samples
        )
        x_samples = np.array(x_samples).reshape(-1, D)

        # update hyper-params for quick re-run tsne
        # note: copy the params since the `dict.update` has side effect
        tsne_hyper_params_for_rerun = deepcopy(tsne_hyper_params)
        tsne_hyper_params_for_rerun.update(early_stop_hyper_params)

        if batch_mode:
            tick = time()
            y_samples = query_blackbox_tsne(
                X=X,
                Y=Y,
                query_idx=selected_idx,
                query_points=x_samples,
                tsne_hyper_params=tsne_hyper_params_for_rerun,
            )
            print(f"[DEBUG] Query-blackbox for {n_samples} points in {time() - tick:.3f} s")
        else:
            y_samples = []
            for i, x_sample in enumerate(x_samples):
                # ask tsne for the corresponding embedding of input sample
                tick = time()
                y_sample = query_blackbox_tsne(
                    X=X,
                    Y=Y,
                    query_idx=selected_idx,
                    query_points=x_sample.reshape(1, D),
                    tsne_hyper_params=tsne_hyper_params_for_rerun,
                )
                print(f"[DEBUG] Query-blackbox for {i+1}th point in {time() - tick:.3f} s")
                y_samples.append(y_sample)
            y_samples = np.array(y_samples).reshape(-1, Y.shape[1])

        # save the sampled points for latter use
        joblib.dump((x_samples, y_samples), log_name_samples)
    return Y, x_samples, y_samples


def query_blackbox_tsne(
    X, Y, query_idx=None, query_points=None, sigma_LD=1.0, tsne_hyper_params={}
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
    assert X.shape[1] == query_points.shape[1], "query_points.shape and X.shape should match"
    assert Y.shape[1] == y_proposals.shape[1], "y_proposals.shape and Y.shape should match"
    X_new = np.concatenate([X, query_points], axis=0)
    Y_new = np.concatenate([Y, y_proposals], axis=0)

    # approximate the original tsne model by a new one with initialized embedding
    # given by the original model
    tsne = modifiedTSNE(init=Y_new, **tsne_hyper_params)
    Y_with_samples = tsne._fit(X_new, skip_num_points=N)

    # debug to see if the original embedding does not change (or change a little bit)
    # TODO: discuss if it is necessary to make the initial Y does not change
    if tsne_hyper_params.get("verbose", 0) > 0:
        diff = Y - Y_with_samples[:N]
        print("[DEBUG] Check if the embedding for N original points does not change")
        print("Norm of diff: ", np.linalg.norm(diff))
        print("Sum square error: ", 0.5 * np.sum(np.dot(diff.T, diff)))

    return Y_with_samples[N:]
