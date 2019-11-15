# minhvu
# 15/11/2019
# explain the samples with a linear models

import numpy as np


def explain_samples(x_samples, y_samples, linear_model, find_rotation=True, use_weight=True):
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

    def rotate_matrix(degree):
        theta = np.radians(degree)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    # greedy search for best rotation angle
    best_score = -np.inf
    best_W = None
    best_angle = 0

    for d in np.arange(0, 180, 1) if find_rotation else [0]:
        R = rotate_matrix(d)
        y_rotated = R.dot(y_samples.T).T

        # x_samples = StandardScaler().fit_transform(x_samples)
        # x_samples = x_samples - x_samples.mean(axis=0, keepdims=True)

        reg = linear_model.fit(x_samples, y_rotated)
        score = reg.score(x_samples, y_rotated)

        if score >= best_score:
            best_score = score
            best_angle = d
            best_W = reg.coef_

    print(f"[Debug]: Best rotation: {best_angle} degree with score = {best_score:.3f}")

    return best_W, best_score, best_angle
