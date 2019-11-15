# minh
# 15/11/2019
################################################################################
# modify TSNE to customize kl_loss and gradient_descent functions
# Note: the `modified_gradient_descent` is not used yet.
# only change the kl_loss function for both exact and barnes_hut method
################################################################################


from time import time

import numpy as np
import sklearn  # note that sklearn does not auto import submodule
import sklearn.manifold  # so we have to import the desired submodules manually
from sklearn.manifold import TSNE
from sklearn.manifold import _barnes_hut_tsne
from scipy import linalg
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform


MACHINE_EPSILON = np.finfo(np.double).eps


def my_kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    # print("[DEBUG] my_kl_divergence")

    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    # grad = np.ndarray((n_samples, n_components), dtype=params.dtype)

    # 15/11/19, minh: hardcode init gradient to zero,
    # to make sure the points with indices below `skip_num_points` do not move.
    # np.ndarray() inits an array with supersmall value like 1e-300,
    # but the accumulate of N points after thousands iterations is considerable.
    grad = np.zeros((n_samples, n_components), dtype=params.dtype)

    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def my_kl_divergence_bh(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    skip_num_points=0,
    verbose=False,
):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2)

    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.

    P : csr sparse matrix, shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized.

    degrees_of_freedom : float
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float (default: 0.5)
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : int
        Verbosity level.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        dof=degrees_of_freedom,
    )

    # 15/11/2019 minh: fix kl_divergence with barnes_hut
    # the original implementation of sklearn does not use `skip_num_points`
    # set gradient of points with indices below `skip_num_points` to zero
    # in order to fix their position in the embedding.
    grad[:skip_num_points] = 0.0

    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad


def modified_gradient_descent(
    objective,
    p0,
    it,
    n_iter,
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    args=None,
    kwargs=None,
):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print(
                    "[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: gradient norm %f. Finished." % (i + 1, grad_norm)
                    )
                break

    return p, error, i


def modifiedTSNE(**kwargs):
    # force using "exact" method to test gradient
    # 15/11/2019: no need to for exact method since we have customize for barnes_hut
    # kwargs.update({"method": "exact"})
    # sklearn.manifold.t_sne._gradient_descent = modified_gradient_descent
    sklearn.manifold.t_sne._kl_divergence = my_kl_divergence
    sklearn.manifold.t_sne._kl_divergence_bh = my_kl_divergence_bh
    tsne = sklearn.manifold.TSNE(**kwargs)
    return tsne


# backtrack: /opt/anaconda3/lib/python3.6/site-packages/sklearn/manifold/t_sne.py
