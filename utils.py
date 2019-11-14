# minhvu
# 13/11/2019
# util functions for ploting

import math

import numpy as np
from matplotlib import pyplot as plt


def scatter_with_samples(Y, y_samples, selected_idx=[], labels=None, out_name="noname00"):
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


def plot_samples(samples, out_name="noname01"):
    """Plot sampled images in a grid of subplots
    """
    img_size = int(math.sqrt(samples.shape[1]))
    n_rows = n_cols = math.ceil(math.sqrt(samples.shape[0]))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for ax in axes.ravel():
        ax.axis("off")

    for img, ax in zip(samples, axes.ravel()):
        ax.imshow(img.reshape(img_size, img_size), cmap="gray_r")

    fig.savefig(out_name)
    plt.close(fig)


def transparent_cmap(cmap, N=255):
    """ Make a transparent color map.
    Hardcode to make the color map be transparent when near zero value
    Credit: https://stackoverflow.com/a/42482371/11722995
    """
    # Copy colormap and set alpha values
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
    return mycmap


def plot_heatmap(W, img=None, out_name="noname00"):
    """Plot heatmap to visualize the weights `W`
    Args:
        W: [2xD], D: dimensionality of input image, which is `img_size x img_size`
    """
    n_targets = len(W)
    img_size = int(math.sqrt(W.shape[1]))
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 4))

    for i, ax in enumerate([axes] if 1 == n_targets else axes.ravel()):
        ax.axis("off")
        if img is not None:
            ax.imshow(img.reshape(img_size, img_size), cmap="gray_r")
        heatmap = ax.imshow(
            W[i, :].reshape(img_size, img_size), cmap=transparent_cmap(plt.cm.inferno)
        )
        fig.colorbar(heatmap, ax=ax)
    fig.savefig(out_name)
    plt.close(fig)
