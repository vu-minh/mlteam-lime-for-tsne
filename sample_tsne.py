# minhvu
# 08/11/2019
# Query-blackbox like function for tsne
# Goal: given an input X, approximate an embedding y without fully rerun tsne

from time import time
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def sample_around(x, sigma=0.1, n_samples=1):
    """Sample a new point around the given datapoint `x`
    `x_sampled = x + epsilon`
    epsilon is a isotropic gaussian noise with variance `sigma`
    """
    epsilon = (sigma ** 2) * np.random.randn(n_samples, len(x.shape))
    return x + epsilon


def tsne_sample_embedded_points(
    X,
    selected_idx,
    n_samples=10,
    sigma_HD=0.1,
    sigma_LD=0.1,
    tsne_hyper_params={},
    early_stop_hyper_params={},
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
    tsne = TSNE(**tsne_hyper_params)
    Y = tsne.fit_transform(X)
    print("[DEBUG] Initial tsne model: ", tsne)

    x_samples = []
    y_samples = []
    for i in range(n_samples):
        # sample a point in HD
        x_sample = sample_around(X[selected_idx], sigma=sigma_HD, n_samples=1)

        # update hyper-params for quick re-run tsne
        tsne_hyper_params.update(early_stop_hyper_params)

        # and ask tsne the corresponding embedding in LD
        tick = time()
        y_sample = query_blackbox_tsne(
            X=X,
            Y=Y,
            query_idx=selected_idx,
            query_points=x_sample,
            sigma_LD=sigma_LD,
            tsne_hyper_params=tsne_hyper_params,
        )
        print(f"[DEBUG] Query-blackbox for {i+1}th point in {time() - tick:.3f} seconds")

        x_samples.append(x_sample)
        y_samples.append(y_sample)

    x_samples = np.array(x_samples).reshape(-1, X.shape[1])
    y_samples = np.array(y_samples).reshape(-1, Y.shape[1])

    return Y, x_samples, y_samples


def query_blackbox_tsne(
    X, Y, query_idx=None, query_points=None, sigma_LD=0.5, tsne_hyper_params={},
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
    Y_with_sample = tsne._fit(X_new, skip_num_points=N)

    # debug to see if the original embedding does not change (or change a little bit)
    # TODO: discuss if it is necessary to make the initial Y does not change
    if tsne_hyper_params.get("verbose", 0) > 0:
        diff = Y - Y_with_sample[:N]
        print("[DEBUG] Check if the embedding for N original points does not change")
        print("Norm of diff: ", np.linalg.norm(diff))
        print("Sum square error: ", 0.5 * np.sum(np.dot(diff.T, diff)))

    return Y_with_sample[N:]


def plot(Y, y_samples, selected_idx=[], labels=None, out_name="noname00"):
    """Plot the original embedding `Y` with new sampled points `y_samples`
    """
    N = Y.shape[0]
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 6))

    # plot the original embedding
    ax0.scatter(Y[:, 0], Y[:, 1], c=labels, alpha=0.5, cmap="jet")
    ax0.scatter(
        Y[selected_idx, 0], Y[selected_idx, 1], marker="s", facecolors="None", edgecolors="b"
    )

    # plot the embedding with new samples
    # Y_new, y_samples = Y_new_with_samples[:N], Y_new_with_samples[N:]
    # print("[DEBUG] Plot: ", Y_new.shape, y_samples.shape)
    # ax1.scatter(Y_new[:, 0], Y_new[:, 1], c=labels, alpha=0.5, cmap="jet")
    ax1.scatter(Y[:, 0], Y[:, 1], c=labels, alpha=0.3, cmap="jet")
    ax1.scatter(
        Y[selected_idx, 0], Y[selected_idx, 1], marker="s", facecolors="None", edgecolors="b"
    )
    ax1.scatter(y_samples[:, 0], y_samples[:, 1], s=32, marker="+", facecolor="r")

    fig.savefig(out_name)
    plt.close(fig)


if __name__ == "__main__":
    n_samples = 25  # number of points to sample
    sigma_HD = 0.5  # larger of Gaussian in HD
    sigma_LD = 1  # larger of Gaussian in LD
    N_max = 200  # maximum number of data points for testing only
    dataset_name = "digits"
    plot_dir = "./plots"
    debug_level = 0

    # TODO check stable of sampling. (e.g. seed 1024 iris, 42 digits)

    # basic params to run tsne the first time
    tsne_hyper_params = dict(perplexity=30, n_iter=1500, random_state=42, verbose=debug_level)

    # to re-run tsne quickly, take the initial embedding as `init` and use the following params
    early_stop_hyper_params = dict(
        early_exaggeration=1,  # disable exaggeration
        n_iter_without_progress=50,
        min_grad_norm=1e-7,
        n_iter=500,
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
    )

    # viz the original embedding with the new sampled points
    out_name = f"{plot_dir}/{dataset_name}_with_sample.png"
    plot(Y, y_samples, selected_idx, labels=labels, out_name=out_name)
