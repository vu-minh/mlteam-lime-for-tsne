# minh and geraldin
# 14/11/2019
# demo explaining tabular data

import os
import math

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression

from sample_tsne import tsne_sample_embedded_points
from explainer import explain_samples, explain_samples_with_cv
from utils import scatter_with_samples, plot_weights
from utils import scatter_for_paper
from utils import load_tabular_dataset


def calculate_weights(y, y_samples):
    """Use the formula q_ij of tsne for weighting the points in `y_samples`
    weight_i = ( 1 + || y - y_i || ^2 ) ^ -1
    """
    dist = 1 + cdist(y_samples, y.reshape(1, -1), "sqeuclidean")  # [n_samples x 1]
    assert dist.shape[0] == y_samples.shape[0]
    return dist ** -1


def filter_by_radius(y_selected, x_samples, y_samples, reject_radius):
    """Filter the samples inside the cycle determined by`reject_radius`
    """
    distances = np.linalg.norm(y_selected - y_samples, axis=1)
    keep_indices = distances <= reject_radius
    return x_samples[keep_indices], y_samples[keep_indices]


def apply_BIR(
    selected_idx,
    x_samples,
    y_samples,
    dataset_name,
    feature_names,
    data_dir,
    BIR_dir,
    lower_bound_lambda=0.0001,
    upper_bound_lambda=3.5,
    nb_lambda=10,
    recompute=True,
):
    """
    """
    # From `y_samples` create the csv file for BIR (as `embedding`)
    y_samples_filename = f"{data_dir}/{dataset_name}_y_samples.csv"
    np.savetxt(y_samples_filename, y_samples.astype(np.float32), delimiter=",", fmt="%.6f")

    # Write `x_samples` to csv file with header (as `dataset`)
    x_samples_filename = f"{data_dir}/{dataset_name}_x_samples.csv"
    np.savetxt(
        x_samples_filename,
        x_samples.astype(np.float32),
        delimiter=",",
        fmt="%.6f",
        header=",".join(feature_names),
        comments="",  # remove the first char (#) in header
    )

    # Prepare the output .Rdata for BIR script and output_directory
    output_directory = f"{data_dir}/{dataset_name}_result"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    output_Rdata = f"{output_directory}.Rdata"

    # pattern to identify the output of BIR
    file_id = f"id{selected_idx}-l{lower_bound_lambda}-u{upper_bound_lambda}-n{nb_lambda}"

    if recompute:
        # Run `Rscript BIR.R embedding.csv dataset.csv output.Rdata`
        lambda_params = f"{lower_bound_lambda} {upper_bound_lambda} {nb_lambda}"
        BIR_script = (
            f"cd {BIR_dir};"
            f"Rscript BIR.R "
            f"../{y_samples_filename} ../{x_samples_filename} ../{output_Rdata} {lambda_params}; "
            f"cd ../"
        )
        print(BIR_script)
        os.system(BIR_script)

        # Run `Rscript from_RData_to_csv.R output.Rdata output_directory`
        convert_script = f"Rscript from_RData_to_csv.R {output_Rdata} {output_directory}/"
        print(convert_script)
        os.system(convert_script)

        # Rename BIR output files
        rename_script = (
            f"mv {output_directory}/BIR_R2.csv {output_directory}/BIR_R2_{file_id}.csv;"
            f"mv {output_directory}/BIR_W.csv {output_directory}/BIR_W_{file_id}.csv;"
            f"mv {output_directory}/BIR_R.csv {output_directory}/BIR_R_{file_id}.csv;"
        )
        print(rename_script)
        os.system(rename_script)

    # Read weights, feature_names, score from `output_directory` (BIR_R2.csv, BIR_W.csv, ...)
    score_data = pd.read_csv(f"{output_directory}/BIR_R2_{file_id}.csv")
    scores = score_data["x"].values

    weight_data = pd.read_csv(f"{output_directory}/BIR_W_{file_id}.csv")
    weights = weight_data[["V1", "V2"]].values.T

    rotation_data = pd.read_csv(f"{output_directory}/BIR_R_{file_id}.csv")
    rotation_mat = rotation_data[["V1", "V2"]].values
    deg = math.degrees(math.acos(rotation_mat[0, 0]))

    return weights, scores, deg


def run_explainer(
    selected_idx,
    linear_model=None,
    use_weights=False,
    reject_radius=0,
    tsne_hyper_params={},
    early_stop_hyper_params={},
    lambda_params={},
):
    """Run the full workflow to samples, do embedding and apply the `linear_model` or BIR
    Args:
        selected_idx: index of a selected point
        linear_model: a sklearn.linear_model for explaining data
            if `linear_model` is None, use BIR
        use_weights: defaults to False: weights the samples by their distance
            to the selected point
        reject_radius: defaults to 0: reject the samples in 2D
            which are outside of the cycle with the given radius.
            If `reject_radius` is zero, do not reject any sample at all.
    """

    # apply the workflow for generating the samples in HD and embed them in LD
    Y, x_samples, y_samples = tsne_sample_embedded_points(
        X,
        selected_idx=selected_idx,
        n_samples=n_samples,
        n_neighbors_SMOTE=n_neighbors_SMOTE,
        tsne_hyper_params=tsne_hyper_params,
        early_stop_hyper_params=early_stop_hyper_params,
        log_dir=log_dir,
        force_recompute=force_recompute,
        batch_mode=False,
    )

    if reject_radius > 0:
        x_samples, y_samples = filter_by_radius(
            Y[selected_idx], x_samples, y_samples, reject_radius
        )

    if use_weights:
        # calculate weights in LD from the selected point and the samples around it
        sample_weights = calculate_weights(Y[selected_idx], y_samples)
        # rescale data according to the `sample_weights`
        # from sklearn.linear_model.base import _rescale_data
        # x_samples, y_samples = _rescale_data(x_samples, y_samples, sample_weights)

        x_samples = np.multiply(sample_weights, x_samples)

    if linear_model is not None:
        # apply the linear model for explaining the sampled points
        W, score, rotation = explain_samples(x_samples, y_samples, linear_model=linear_model)
        title = f"Best score $R^2$ = {score:.3f}, best rotation = {rotation:.0f} deg"
    else:
        # apply BIR to obtain W and scores for 2 axes
        W, scores, rotation = apply_BIR(
            selected_idx,
            x_samples,
            y_samples,
            dataset_name,
            feature_names,
            data_dir,
            BIR_dir,
            recompute=True,  # for ploting the figures without re-run BIR
            **lambda_params,
        )
        title = (
            f"Best $R^2$ for 1st axis {scores[0]:.3f} and for 2nd axis {scores[1]:.3f}"
            f"\nRotation {rotation:.3f} deg"
        )

    # viz the original embedding with the new sampled points
    out_name_prefix = (
        f"{plot_dir}/id{selected_idx}-n{n_samples}"
        f"-{'BIR' if linear_model is None else 'LM'}"
        f"-r{reject_radius}"
        f"-w{1 if use_weights else 0}"
    )
    out_name_Y = f"{out_name_prefix}_scatter.png"
    out_name_sample = f"{out_name_prefix}_samples.png"
    scatter_for_paper(
        X=X,
        Y=Y,
        W=W,
        y_samples=y_samples,
        selected_idx=selected_idx,
        rot_deg=rotation,
        out_name=out_name_sample,
    )

    # visualize the weights of the linear model
    # (show contribution of the most important features)
    lambda_param_str = f"l{lambda_params['lower_bound_lambda']}-u{lambda_params['upper_bound_lambda']}-n{lambda_params['nb_lambda']}"
    out_name_W = f"{out_name_prefix}_explanation_{lambda_param_str}.png"
    plot_weights(W, feature_names, out_name=out_name_W, left_margin=0.2)
    print("[DEBUG] Chart title: ", title)


if __name__ == "__main__":
    dataset_name = "country"
    data_dir = "dataset"
    BIR_dir = "./BIR"
    log_dir = f"./var/{dataset_name}"
    plot_dir = f"./plots/{dataset_name}"
    for a_dir in [plot_dir, log_dir]:
        if not os.path.exists(a_dir):
            os.mkdir(a_dir)

    # define the variables for the dataset name, number of samples, ...
    # these global variables can be overrided by the new values loaded from the config
    n_samples = 100  # number of points to sample
    n_neighbors_SMOTE = 10  # number of neighbors to take for SMOTE over sampling
    seed = 42  # for reproducing
    debug_level = 0  # verbose in tsne, 0 to disable
    force_recompute = False  # use pre-calculated embedding and samples or recompute them

    # load config for tsne hyper-params and particular config for the selected points
    from config_selected_points import config_selected_points

    config = config_selected_points.get(dataset_name, {})
    # override some global config
    if config:
        n_samples = config.get("n_samples", 100)
        n_neighbors_SMOTE = config.get("n_neighbors_SMOTE", 10)
        seed = config.get("seed", 42)

    # basic params to run tsne the first time
    tsne_hyper_params = config.get(
        "tsne_hyper_params",
        dict(method="exact", perplexity=10, n_iter=1000, random_state=42, verbose=debug_level),
    )

    # to re-run tsne quickly, take the initial embedding as `init` and use the following params
    early_stop_hyper_params = config.get(
        "early_stop_hyper_params",
        dict(
            early_exaggeration=1,  # disable exaggeration
            n_iter_without_progress=100,
            min_grad_norm=1e-7,
            n_iter=500,  # increase it to test stability
            verbose=debug_level,
        ),
    )

    # set numpy random seed for reproducing
    np.random.seed(seed)

    # load the chosen dataset
    X, labels, feature_names = load_tabular_dataset(dataset_name, standardize=True)

    config_selected_points = config.get("selected_points", {np.random.randint(X.shape[0]): {}})
    for selected_idx, config_selected_point in config_selected_points.items():
        selected_idx = int(selected_idx)
        print("[DEBUG] selected point: ", selected_idx, labels[selected_idx])

        # run full workflow with a `linear_model` or with BIR if the `linear_model` is None
        linear_model = None  # ElasticNet()
        reject_radius = config_selected_point.get("reject_radius", 5)
        use_weights = config_selected_point.get("use_weights", False)
        lambda_params = config_selected_point.get(
            "lambda_params",
            dict(lower_bound_lambda=0.01, upper_bound_lambda=1.0, nb_lambda=10),
        )
        run_explainer(
            selected_idx=selected_idx,
            linear_model=linear_model,
            use_weights=use_weights,
            reject_radius=reject_radius,
            tsne_hyper_params=tsne_hyper_params,
            early_stop_hyper_params=early_stop_hyper_params,
            lambda_params=lambda_params,
        )
