# minhvu
# 15/11/2019
# explain the samples with a linear models

import numpy as np
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

from sklearn.linear_model import ElasticNet, MultiTaskElasticNetCV


def rotate_matrix(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def optimize_linear_model(X, Y, linear_model, space, random_state):
    @use_named_args(space)
    def objective(rotation, alpha, l1_ratio):
        R = rotate_matrix(rotation)
        Y_rotated = R.dot(Y.T).T

        linear_model.set_params(alpha=alpha, l1_ratio=l1_ratio)
        linear_model.fit(X, Y_rotated)

        r_square_score = linear_model.score(X, Y_rotated)
        print("score: ", r_square_score)
        return 1.0 - r_square_score

    res_gp = gp_minimize(objective, space, n_calls=50, random_state=random_state)
    return res_gp.x, res_gp.fun


def explain_samples(x_samples, y_samples, linear_model):
    linear_model = ElasticNet()

    space = [
        Real(0, 90, name="rotation"),
        Real(0.1, 2.0, name="alpha"),
        Real(0.1, 1.0, name="l1_ratio"),
    ]
    (rotation, alpha, l1_ratio), score = optimize_linear_model(
        x_samples, y_samples, linear_model, space, random_state=42
    )
    best_score = 1.0 - score

    print(f"[DEBGU]: Best rotation: {rotation} degree with score = {best_score:.3f}")
    print(f"[DEBUG]: alpha = {alpha:.3f}, l1_ratio = {l1_ratio:.3f}")

    R = rotate_matrix(rotation)
    y_rotated = R.dot(y_samples.T).T
    linear_model.set_params(alpha=alpha, l1_ratio=l1_ratio)
    linear_model.fit(x_samples, y_rotated)

    best_W = linear_model.coef_
    return best_W, best_score, rotation


def explain_samples_with_cv(
    x_samples, y_samples, linear_model, find_rotation=True, use_weight=True
):
    """Naive explainer by a linear regression (LR) model.
    Note that, we do not use the intercept, so we should standardize the input.
    Args:
        x_samples: [n_samples x D]
        y_samples: [n_samples x 2]
        find_rotation: bool default to True: find the best rotation of `y_samples`
            that minimizes the `scores` of the linear model.
    Returns:
        weights of LR model (representing the importance of each pixel)
    """

    # greedy search for best rotation angle
    best_score = -np.inf
    best_W = None
    best_angle = 0
    best_alpha = None
    best_l1_ratio = None

    linear_model = MultiTaskElasticNetCV(cv=5)

    for d in np.arange(0, 90, 1) if find_rotation else [0]:
        R = rotate_matrix(d)
        y_rotated = R.dot(y_samples.T).T

        reg = linear_model.fit(x_samples, y_rotated)
        score = reg.score(x_samples, y_rotated)

        if score >= best_score:
            best_score = score
            best_angle = d
            best_W = reg.coef_
            best_alpha = reg.alpha_
            best_l1_ratio = reg.l1_ratio_

    print(f"[DEBGU]: Best rotation: {best_angle} degree with score = {best_score:.3f}")
    print(f"[DEBUG]: alpha = {best_alpha:.3f}, l1_ratio = {best_l1_ratio:.3f}")

    return best_W, best_score, best_angle
