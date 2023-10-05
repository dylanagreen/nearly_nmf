#!/usr/bin/env python
import numpy as np
import numpy.typing as npt

nan_eps = 1e-6
def shift_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike,
            W_start: npt.ArrayLike, n_iter: int = 500, update_H: bool = True,
            update_W: bool = True, return_chi_2: bool = False) ->  tuple[npt.NDArray, npt.NDArray]:
    """Fit NMF templates to noisy, possibly negative, data with weights using the "shift-NMF" algorithm.

    WARNING: This function will do no sanity checking of inputs. It is highly recommended to use `fit_NMF`
    to fit templates instead of calling this function directly.

    Parameters
    ----------
    X : array_like
        Input data of shape (n_dimensions, n_observations).
    V : array_like
        Input weights of shape (n_dimensions, n_observations).
    H_start : array_like
        Starting point for the H matrix. Must have shape (n_templates, n_observations).
    W_start : array_like
        Starting point for the W matrix. Must have shape (n_dimensions, n_templates).
    n_iter : int, optional
        Number of fitting iterations to run. Defaults to 500.
    update_H : bool, optional
        Whether to update H when running the iteration. Set to False
        to fit only templates with the given coefficients. Defaults to True.
    update_W : bool, optional
        Whether to update W when running the iteration. Set to False
        to fit only coefficients with the given templates. Defaults to True.
    return_chi_2 : bool, optional
        Whether to track and return the chi^2 history of the fit. This
        involves computing a matrix norm, so will slightly slow the
        fit depending on data size. Defaults to False.

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates,
        n_observations). Each column represents the coefficients
        corresponding to a column of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates).
        Each column represents one template.
    chi_2 : numpy.ndarray, optional
        The chi^2 history of the fit.  Only returned if `return_chi_2` is True.
    """
    # Copy H and W to avoid mutating the inputs
    X, V = np.asarray(X), np.asarray(V)
    H, W = np.array(H_start, copy=True), np.array(W_start, copy=True)

    chi_2 = []
    # Only shift if the lowest value of X is negative, otherwise
    # we can ignore the shifting
    shift = np.min(X)
    if shift < 0:
        # Since the shift is negative we need to subtract it here rather
        # than add it to shift X so the minimum is 0
        X = X - shift
    else:
        shift = 0

    # The initial chi^2 pre fitting
    if return_chi_2:
        c2 = np.sum((np.sqrt(V) * (X - (W @ H - shift))) ** 2)
        chi_2.append(c2)

    V_X = V * X # Weighted X, outside the loop for efficiency
    for _ in range(n_iter):
        # H Step
        if update_H:
            H = H * (W.T @ (V_X)) / (W.T @ (V * (W @ H - shift)))

            # Here to ensure that nans are converted to zeros.
            # If the weights are set to 0 we might end up with
            # a division by 0 and a corresponding nan/inf that needs
            # to be handled correctly
            H = np.nan_to_num(H, nan=nan_eps, posinf=nan_eps)

        # W Step
        if update_W:
            W = W * ((V_X) @ H.T) / ((V * (W @ H - shift)) @ H.T)
            W = np.nan_to_num(W, nan=nan_eps, posinf=nan_eps)

        if return_chi_2:
            c2 = np.sum((np.sqrt(V) * (X - (W @ H - shift))) ** 2)
            chi_2.append(c2)

    if return_chi_2:
        return H, W, chi_2
    else:
        return H, W

def split_pos_neg(A: npt.ArrayLike):
    """Splits a matrix into its positive and negative elements, with all other values set to 0.

    Parameters
    ----------
    A : array_like
        Input array of any shape.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as A, with negative or zero elements set to 0.
    numpy.ndarray
        Array of the same shape as A, with positive or zero elements set to 0.
    """
    A = np.asarray(A)
    return (np.abs(A) + A) / 2, (np.abs(A) - A) / 2


def nearly_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike,
            W_start: npt.ArrayLike, n_iter: int = 500, update_H: bool = True,
            update_W: bool = True, return_chi_2: bool = False) ->  tuple[npt.NDArray, npt.NDArray]:
    """Fit NMF templates to noisy, possibly negative, data with weights using the "nearly-NMF" algorithm.

    WARNING: This function will do no sanity checking of inputs. It is highly recommended to use `fit_NMF`
    to fit templates instead of calling this function directly.
    Parameters
    ----------
    X : array_like
        Input data of shape (n_dimensions, n_observations).
    V : array_like
        Input weights of shape (n_dimensions, n_observations).
    H_start : array_like
        Starting point for the H matrix. Must have shape (n_templates, n_observations).
    W_start : array_like
        Starting point for the W matrix. Must have shape (n_dimensions, n_templates).
    n_iter : int, optional
        Number of fitting iterations to run. Defaults to 500.
    update_H : bool, optional
        Whether to update H when running the iteration. Set to False
        to fit only templates with the given coefficients. Defaults to True.
    update_W : bool, optional
        Whether to update W when running the iteration. Set to False
        to fit only coefficients with the given templates. Defaults to True.
    return_chi_2 : bool, optional
        Whether to track and return the chi^2 history of the fit. This
        involves computing a matrix norm, so will slightly slow the
        fit depending on data size. Defaults to False.

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates,
        n_observations). Each column represents the coefficients
        corresponding to a column of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates).
        Each column represents one template.
    chi_2 : numpy.ndarray, optional
        The chi^2 history of the fit.  Only returned if `return_chi_2` is True.
    """
    # Copy H and W to avoid mutating the inputs
    X, V = np.asarray(X), np.asarray(V)
    H, W = np.array(H_start, copy=True), np.array(W_start, copy=True)

    chi_2 = []
    # The initial chi^2 pre fitting
    if return_chi_2:
        c2 = np.sum((np.sqrt(V) * (X - W @ H)) ** 2)
        chi_2.append(c2)
    # Precomputing some values for efficiency
    V_X = V * X
    for j in range(n_iter):
        # H-step
        if update_H:
            W_VX = W.T @ V_X
            W_VX_pos, W_VX_neg = split_pos_neg(W_VX)

            H = H * (W_VX_pos) / (W.T @ (V * (W @ H)) + W_VX_neg)
            H = np.nan_to_num(H, nan=nan_eps, posinf=nan_eps)
        # W-step
        if update_W:
            V_XH = V_X @ H.T
            V_XH_pos, V_XH_neg = split_pos_neg(V_XH)

            W = W * (V_XH_pos) / ((V * (W @ H)) @ H.T + V_XH_neg)
            W = np.nan_to_num(W, nan=nan_eps, posinf=nan_eps)

        if return_chi_2:
            c2 = np.sum((np.sqrt(V) * (X - W @ H)) ** 2)
            chi_2.append(c2)

    if return_chi_2:
        return H, W, chi_2
    else:
        return H, W

def fit_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike = None,
            W_start: npt.ArrayLike = None, n_templates: int = 2,
            n_iter: int = 500, update_H: bool = True,
            update_W: bool = True, algorithm: str = "shift",
            return_chi_2: bool = False) ->  tuple[npt.NDArray, npt.NDArray]:
    """Fit NMF templates to noisy, possibly negative, data with weights using the specified algorithm.

    Parameters
    ----------
    X : array_like
        Input data of shape (n_dimensions, n_observations).
    V : array_like
        Input weights of shape (n_dimensions, n_observations).
    H_start : array_like, optional
        Starting point for the H matrix. Defaults to a matrix of random
        normal variables with a mean of 0 and a sigma of 2. If provided
        must have shape (n_templates, n_observations).
    W_start : array_like, optional
        Starting point for the W matrix. Defaults to a matrix of random
        normal variables with a mean of 0 and a sigma of 2. If provided
        must have shape (n_dimensions, n_templates).
    n_templates : int, optional
        Number of templates to fit. Not necessary when providing
        a starting matrix for H or W. Defaults to 2.
    n_iter : int, optional
        Number of fitting iterations to run. Defaults to 500.
    update_H : bool, optional
        Whether to update H when running the iteration. Set to False
        to fit only templates with the given coefficients. Defaults to True.
    update_W : bool, optional
        Whether to update W when running the iteration. Set to False
        to fit only coefficients with the given templates. Defaults to True.
    algorithm : {"shift", "nearly"}, optional
        The algorithm to use to do the fit.
            * "shift" : Uses the shift-NMF algorithm, where all data is shifted
            to the nonnegative half plane and NMF is fitted with a fixed
            template/coefficient pair accounting for the shift.
            * "nearly" : Uses the nearly-NMF algorithm, where the positive
            and negative components of the input data are seperated and accounted
            for separately in the update rules.
    return_chi_2 : bool, optional
        Whether to track and return the chi^2 history of the fit. This
        involves computing a matrix norm, so will slightly slow the
        fit depending on data size. Defaults to False.

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates,
        n_observations). Each column represents the coefficients
        corresponding to a column of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates).
        Each column represents one template.
    chi_2 : numpy.ndarray, optional
        The chi^2 history of the fit.  Only returned if `return_chi_2` is True.
    """

    if (H_start is not None) and (W_start is not None):
        assert H_start.shape[0] == W_start.shape[1], "Number of templates does not match between H and W"
    assert (update_H or update_W), "At least one of update_H or update_W must be True"
    if H_start is not None: n_templates = H_start.shape[0]
    elif W_start is not None: n_templates = W_start.shape[1]

    # Size of the coefficients and templates respectively
    # to ensure we use the same size everywhere
    H_shape = (n_templates, X.shape[1])
    W_shape = (X.shape[0], n_templates)
    rng = np.random.default_rng(100921)

    # Randomly initialize the H and W matrices if necessary.
    if H_start is not None:
        H = np.asarray(H_start)
    else:
        H = rng.uniform(0, 2, H_shape)


    if W_start is not None:
        W = np.asarray(W_start)
    else:
        W = rng.uniform(0, 2, W_shape)

    if algorithm == "shift":
        to_return = shift_NMF(X, V, H, W, n_iter, update_H, update_W, return_chi_2)
    elif algorithm == "nearly":
        to_return = nearly_NMF(X, V, H, W, n_iter, update_H, update_W, return_chi_2)
    else:
        print("Algorithm not found, aborting!")
        return

    return to_return


class NMF:
    def __init__(self, X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike = None,
                 W_start: npt.ArrayLike = None, n_templates: int = 2, n_iter: int = 500,
                 algorithm: str = "shift", return_chi_2: bool = False):
        """An NMF model object. This object holds all of the relevant NMF algorithmic data, namely
        fitted coefficients and templates/basis vectors for its training dataset.

        Parameters
        ----------
        X : array_like
            Input data of shape (n_dimensions, n_observations).
        V : array_like
            Input weights of shape (n_dimensions, n_observations).
        H_start : array_like, optional
            Starting point for the H matrix. Defaults to a matrix of random
            normal variables with a mean of 0 and a sigma of 2. If provided
            must have shape (n_templates, n_observations).
        W_start : array_like, optional
            Starting point for the W matrix. Defaults to a matrix of random
            normal variables with a mean of 0 and a sigma of 2. If provided
            must have shape (n_dimensions, n_templates).
        n_templates : int, optional
            Number of templates to fit. Not necessary when providing
            a starting matrix for H or W. Defaults to 2.
        n_iter : int, optional
            Number of fitting iterations to run. Defaults to 500.
        return_chi_2 : bool, optional
            Whether to track and return the chi^2 history of the fit. This
            involves computing a matrix norm, so will slightly slow the
            fit depending on data size. Defaults to False.

        """
        self.X, self.V = np.asarray(X), np.asarray(V)

        if (H_start is not None) and (W_start is not None):
            assert H_start.shape[0] == W_start.shape[1], "Number of templates does not match between H and W"
        if H_start is not None: self.n_templates = H_start.shape[0]
        elif W_start is not None: self.n_templates = W_start.shape[1]
        else: self.n_templates = n_templates

        # Size of the coefficients and templates respectively
        # to ensure we use the same size everywhere
        H_shape = (n_templates, X.shape[1])
        W_shape = (X.shape[0], n_templates)
        self.rng = np.random.default_rng(100921)

        # Randomly initialize the H and W matrices if necessary.
        if H_start is not None:
            self.H = np.asarray(H_start)
        else:
            self.H = self.rng.uniform(0, 2, H_shape)

        if W_start is not None:
            self.W = np.asarray(W_start)
        else:
            self.W = self.rng.uniform(0, 2, W_shape)

        # Internally store which fitting function we'll be using since the
        # object initialization has done sanity checking.
        self.fit_NMF = shift_NMF if algorithm == "shift" else nearly_NMF
        self.return_chi_2 = return_chi_2

        self.chi_2 = []
        self.n_iter = n_iter

    def fit(self):
        """Fit this NMF object to noisy, possibly negative, data with weights.
        """
        if self.return_chi_2:
            self.H, self.W, self.chi_2 = self.fit_NMF(self.X, self.V, self.H, self.W, n_iter=self.n_iter, return_chi_2=self.return_chi_2)
        else:
            self.H, self.W = self.fit_NMF(self.X, self.V, self.H, self.W, n_iter=self.n_iter)

    def predict(self) ->  npt.NDArray:
        """Generate a reconstruction of the original input data using the stored factorization.

        Returns
        -------
         numpy.ndarray
            A reconstruction of the original input dataset, with same shape and
            size.
        """
        return self.W @ self.H

    def fit_coeffs(self, X: npt.ArrayLike, V: npt.ArrayLike) -> npt.NDArray:
        """Generate coefficients fitting this object's NMF templates
        to noisy, possibly negative, data with weights.

        Parameters
        ----------
        X : array_like
            Input data of shape (n_dimensions, n_observations).
        V : array_like
            Input weights of shape (n_dimensions, n_observations).

        Returns
        -------
        H : numpy.ndarray
            The fitted NMF coefficients with shape (n_templates,
            n_observations) for the objects internal templates.
            Each column represents the coefficients corresponding
            to a column of X.
        """
        return self.fit_NMF(X, V, self.H, self.W, n_iter=self.n_iter, update_W=False)[0]


