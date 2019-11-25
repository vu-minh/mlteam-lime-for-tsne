# minhvu
# 13/11/2019
# util functions for ploting and loading dataset

import math
import joblib
import string

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine, load_iris, load_boston
from sklearn.preprocessing import StandardScaler, Normalizer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker  # for customizing number of ticks


plt.rcParams.update({"font.size": 18})


def rotate_matrix(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def clean_feature_names(feature_names):
    def _clean_words(words):
        return "".join(w for w in words if w.isalnum() or w in string.whitespace)

    return [_clean_words(feature_name) for feature_name in feature_names]


def load_country():
    data = joblib.load("./dataset/country.dat")
    fix_feature_names = [
        "Pop growth",
        "Pop growth 2004",
        "Price index",
        "Carbon Dioxide 2003",
        "Export 1990",
        "Export 2004",
        "Elec 2003",
        "GDP",
        "GDP PPP",
        "GDP pc",
        "GDP pc growth rate",
        "Fem Econo Rate",
        "Fem Econo 1990",
        "Fem Econo 2004",
        "Health Exp",
        "Babies",
        "Internet 1990",
        "Import 1990",
        "Import 2004",
        "Tertiary female ratio",
        "Babies immunized",
        "Manufactured Exp 2004",
        "Foreign invest 2004",
        "Military 2004",
        "Public Health 2003",
        "Private Health 2003",
        "Primary export 2004",
        "Public Health",
        "Refugees asylum",
        "Refugees origin",
        "Armed forces",
        "Parliament Seats Women",
        "Female Male income",
        "House women 2006",
        "Pop 1975",
        "Pop 2004",
        "Pop 2015",
        "Tuberculosis detected",
        "Tuberculosis cured 2004",
        "Trad fuel",
        "ODA pc donnor 2004",
        "ODA to least dev 1990",
        "ODA to least dev 2004",
        "ODA received",
        "ODA received pc",
    ]
    return {
        "data": data["X"],
        "target": data["country_names"],
        "feature_names": fix_feature_names
        # clean_feature_names(data["indicator_descriptions"]),
    }


def load_automobile():
    # Ref: https://www.kaggle.com/toramky/automobile-dataset
    return joblib.load("./dataset/Automobile.pkl")


def load_cars_dataset():
    # UCI cars dataset: https://archive.ics.uci.edu/ml/datasets/automobile
    df = pd.read_csv("./dataset/cars1985.csv")
    print(df.describe())

    instance_names = df["Vehicle Name"].tolist()
    print(instance_names)

    column_names = list(df.columns)
    print(column_names)

    # The "Engine Size (l)" column is string. Convert it to float
    df["Engine Size (l)"] = df["Engine Size (l)"].str.replace(",", "").astype(float)

    # Fill NAN by the average value of the column
    df.fillna(df.mean(), inplace=True)

    # Get all columns except the first one and convert to numpy array
    data = df.loc[:, df.columns != "Vehicle Name"].to_numpy()
    print(data.shape)

    return {
        "data": data,
        "target": instance_names,
        "feature_names": column_names[1:],  # remove first column of vehicle name
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
        "cars1985": load_cars_dataset,
        "automobile": load_automobile,
        "wine": load_wine,
        "iris": load_iris,
        "boston": load_boston,
    }[dataset_name]

    data = load_func()
    X, label_names, feature_names = (
        data["data"],
        data["target"],
        data["feature_names"],
    )
    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, label_names, feature_names


def scatter_with_samples(
    Y,
    y_samples=None,
    selected_idx=[],
    labels=None,
    texts=None,
    text_length=3,
    rot_deg=0,
    out_name="noname00",
):
    """Plot the original embedding `Y` with new sampled points `y_samples`
    Can plot with numeric labels of points if `labels` is not None,
    and/or with text if `texts` is not None.
    The number of character of text is set by `text_length`
    """
    N = Y.shape[0]
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 6))
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")

    # plot the original embedding
    ax0.scatter(Y[:, 0], Y[:, 1], c=labels, alpha=0.5, cmap="jet")
    ax0.scatter(
        Y[selected_idx, 0], Y[selected_idx, 1], marker="s", facecolors="None", edgecolors="b"
    )
    if texts is not None:
        for i, text in enumerate(texts):
            ax0.text(x=Y[i, 0], y=Y[i, 1], s=str(text)[:text_length])

    # plot the embedding with new samples
    ax1.scatter(Y[:, 0], Y[:, 1], c=labels, alpha=0.3, cmap="jet")
    ax1.scatter(
        Y[selected_idx, 0], Y[selected_idx, 1], marker="s", facecolors="None", edgecolors="b"
    )
    if y_samples is not None:
        ax1.scatter(y_samples[:, 0], y_samples[:, 1], s=32, marker="+", facecolor="r")

    # plot 2 orthogonal axes
    plot_perpendicular_lines(ax1, Y[selected_idx], rot_deg)

    fig.savefig(out_name)
    plt.close(fig)


def plot_perpendicular_lines(ax, p0, rot_deg=0, axis_length=15):
    """plot two perpendicular axes with center `p0`
        and being rotated `rot_deg` degree
    """
    x0, y0 = p0
    linestyle = "--"
    text_offset = 0.1 * axis_length

    # note: rotation is counter-clockwise
    R = rotate_matrix(rot_deg)
    coor_axes = axis_length * R @ np.eye(2)
    axes_colors = ["indigo", "green"]  # ["#1f77b4", "#ff7f0e"]
    for i, ([x, y], color) in enumerate(zip(coor_axes, axes_colors)):
        ax.arrow(
            x=x0,
            y=y0,
            dx=x,
            dy=y,
            head_width=0.075 * axis_length,
            head_length=0.075 * axis_length,
            edgecolor=color,
            linestyle=linestyle,
        )
        ax.text(
            x=x0 + x + text_offset,
            y=y0 + y + text_offset,
            s=f"W{i+1}",
            color=color,
            ha="center",
            fontsize=14,
            # weight="bold",
            zorder=99,
        )
        ax.plot([x0, x0 - x], [y0, y0 - y], color=color, linestyle=linestyle)


def scatter_samples_with_rotated_axes(y, y_samples, rot_deg=0, out_name="noname02", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_aspect("equal")

    x_min, y_min = np.min(y_samples[:, 0]), np.min(y_samples[:, 1])
    x_max, y_max = np.max(y_samples[:, 0]), np.max(y_samples[:, 1])
    diff = 1.2 * max(abs(x_max - x_min), abs(y_max - y_min))
    ax.set_xlim(y[0] - diff, y[0] + diff)
    ax.set_ylim(y[1] - diff, y[1] + diff)

    ax.scatter(y[0], y[1], marker="s", facecolors="None", edgecolors="b", zorder=98)
    ax.scatter(y_samples[:, 0], y_samples[:, 1], s=32, marker="+", facecolor="r")
    plot_perpendicular_lines(ax, y, rot_deg, axis_length=0.75 * diff)

    if ax is None and out_name:
        plt.tight_layout()
        fig.savefig(out_name)
        plt.close(fig)


def scatter_embedding_with_samples_and_rotated_axes(
    X, Y, W, y_samples, selected_idx, texts=None, rot_deg=0, out_name="noname03", text_length=0
):
    """ show subplots ax0 | ax1, in which
        ax0 shows the embedding `Y` with the selected point `selected_idx` and the `y_samples`
        ax1 zooms in the samples and shows the rotated coordinate axes
        ax0 shows text for each point if texts is given and `text_length` > 0
    """
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 6))
    ax0.set_aspect("equal")
    ax1.set_aspect("equal")

    y0 = Y[selected_idx]

    # calculate prediction error of the linear model
    Y_target = Y - Y[selected_idx]
    Y_target -= Y_target.mean(axis=1)
    errors = prediction_error(X, W, Y_target, rot_deg)

    # plot the embedding
    sc = ax0.scatter(Y[:, 0], Y[:, 1], c=errors, cmap="RdBu", alpha=0.5)
    ax0.scatter(y0[0], y0[1], zorder=99, marker="s", facecolors="None", edgecolors="b")
    # plot the samples
    ax0.scatter(y_samples[:, 0], y_samples[:, 1], s=32, marker="+", facecolor="r")
    fig.colorbar(sc, ax=ax0)

    # show text
    if text_length > 0 and texts is not None:
        for i, text in enumerate(texts):
            ax.text(x=Y[i, 0], y=Y[i, 1], s=str(text)[:text_length])

    # show title
    if texts is not None:
        title = f"Selected point: {texts[selected_idx]}"
        ax0.set_title(title)

    # in the other subplot, plot the samples with rotated axes
    scatter_samples_with_rotated_axes(y0, y_samples, rot_deg, ax=ax1)
    ax1.set_title(f"Best rotation: {rot_deg:.0f} deg")

    plt.tight_layout()
    fig.savefig(out_name)
    plt.close(fig)


def scatter_for_paper(X, Y, W, y_samples, selected_idx, rot_deg=0, out_name="noname04"):
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 6))
    y0 = Y[selected_idx]

    # plot the original embedding with samples and rotated axes
    _plot_embdding_with_inset_zoom(ax0, Y, y0, y_samples, rot_deg)

    # plot prediction error of the linear model
    errors = prediction_error(X, W, Y - Y[selected_idx], rot_deg)
    _plot_embedding_with_error(ax1, Y, y0, y_samples, rot_deg, errors)

    plt.tight_layout()
    fig.savefig(out_name)
    plt.close(fig)


def _plot_embedding_with_error(ax, Y, y0, y_samples, rot_deg, errors):
    # scatter plot with error of each point represented by color
    ax.set_aspect("equal")
    scatter = ax.scatter(
        Y[:, 0],
        Y[:, 1],
        s=64,
        c=errors,
        alpha=0.85,
        zorder=10,
        cmap="Blues_r",
        linewidth=1.0,
        edgecolor="b",
    )
    ax.scatter(
        y_samples[:, 0], y_samples[:, 1], s=32, marker="+", facecolor="r", zorder=1, alpha=0.5
    )
    ax.scatter(y0[0], y0[1], marker="s", facecolors="None", edgecolors="b", zorder=99)
    plot_perpendicular_lines(ax, y0, rot_deg, axis_length=15)

    # colorbar
    cbaxes = inset_axes(ax, width="50%", height="5%", loc=2)
    cb = plt.colorbar(scatter, cax=cbaxes, orientation="horizontal")
    cb.ax.xaxis.set_major_locator(ticker.AutoLocator())
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()


def _plot_embdding_with_inset_zoom(ax, Y, y0, y_samples, rot_deg):
    # plot the embedding and samples in the first plot
    ax.set_aspect("equal")
    ax.scatter(
        Y[:, 0], Y[:, 1], s=64, color="#D3D3D3", alpha=1.0, linewidth=1.0, edgecolor="#808080",
    )
    ax.scatter(y_samples[:, 0], y_samples[:, 1], s=32, marker="+", facecolor="r")
    ax.scatter(y0[0], y0[1], marker="s", facecolors="None", edgecolors="b", zorder=99)

    # limit for zoom-in region
    x_min, y_min = np.min(y_samples[:, 0]), np.min(y_samples[:, 1])
    x_max, y_max = np.max(y_samples[:, 0]), np.max(y_samples[:, 1])
    diff = 0.8 * max(abs(x_max - x_min), abs(y_max - y_min))

    # inset zoom in
    axins = ax.inset_axes([0.025, 0.525, 0.45, 0.45])
    axins.set_aspect("equal")
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)
    axins.set_xlim(y0[0] - diff, y0[0] + diff)
    axins.set_ylim(y0[1] - diff, y0[1] + diff)
    ax.indicate_inset_zoom(axins)

    axins.scatter(
        Y[:, 0], Y[:, 1], s=64, color="#D3D3D3", alpha=1.0, linewidth=1.0, edgecolor="#808080",
    )
    axins.scatter(y_samples[:, 0], y_samples[:, 1], s=64, marker="+", facecolor="r")
    axins.scatter(y0[0], y0[1], marker="s", facecolors="None", edgecolors="b", zorder=99)
    plot_perpendicular_lines(axins, y0, rot_deg, axis_length=0.75 * diff)


def prediction_error(X, W, Y, rot_deg):
    R = rotate_matrix(rot_deg)
    diff = X @ W.T - Y @ R
    mse = np.linalg.norm(diff, axis=1) / X.shape[0]
    return mse


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


def plot_heatmap(W, img=None, title="", out_name="noname00"):
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

    plt.suptitle(title)
    fig.savefig(out_name)
    plt.close(fig)


def plot_weights(
    W, feature_names=None, titles=["", ""], left_margin="auto", out_name="", filter_zeros=True,
):
    """Plot importance (weights) of each feature
    Args:
        W: [2xD]
        feature_names: [D,]
    """
    assert W is not None, "Error with linear model!"
    max_text_length = max(list(map(len, feature_names)))

    if filter_zeros:
        keep_indices = []
        for i in range(W.shape[1]):
            if W[0, i] != 0.0 or W[1, i] != 0.0:
                keep_indices.append(i)
        W = W[:, keep_indices]
        feature_names = np.array(
            [
                f"{s:>{max_text_length+2}}"
                for i, s in enumerate(feature_names)
                if i in keep_indices
            ]
        )

    if feature_names is None:
        feature_names = [f"f{i+1}" for i in range(W.shape[1])]

    n_cols = W.shape[0]
    fig, axes = plt.subplots(
        1, n_cols, figsize=(n_cols * 5, W.shape[1] * 0.25 + 2), sharey=True
    )
    for ax, weights, title in zip(axes.ravel(), W, titles):
        y_pos = np.arange(len(feature_names))
        ax.barh(
            y_pos,
            weights,
            height=0.65,
            align="center",
            color=list(map(lambda w: "#2ca02c" if w > 0 else "#d62728", weights.tolist())),
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=18)
        ax.set_title(title, fontsize=18)
        # ax.set_xlabel("Importance of features")

    if isinstance(left_margin, float):
        fig.subplots_adjust(left=left_margin)
    # fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_name)
    plt.close(fig)
