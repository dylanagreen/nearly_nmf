import numpy as np
import numpy.typing as npt

def fit_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike = None,
            W_start: npt.ArrayLike = None, n_templates: int = 2,
            n_iter: int = 500, update_H: bool = True,
            update_W: bool = True) ->  tuple[npt.NDArray, npt.NDArray]:
    """Fit NMF templates to noisy, possibly negative, data with weights.

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

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates,
        n_observations). Each column represents the coefficients
        corresponding to a column of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates).
        Each column represents one template.
    """
    X, V = np.asarray(X), np.asarray(V)

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

    # Only shift if the lowest value of X is negative, otherwise
    # we can ignore the shifting
    shift = np.min(X)
    if shift < 0:
        # Since the shift is negative we need to subtract it here rather
        # than add it to shift X so the minimum is 0
        X = X - shift
    else:
        shift = 0
    V_X = V * X # Weighted X, outside the loop for efficiency
    for _ in range(n_iter):
        # H Step
        if update_H:
            H = H * (W.T @ (V_X)) / (W.T @ (V * (W @ H - shift)))

            # Here to ensure that nans are converted to zeros.
            # If the weights are set to 0 we might end up with
            # a division by 0 and a corresponding nan/inf that needs
            # to be handled correctly
            H = np.nan_to_num(H, nan=0, posinf=0)

        # W Step
        if update_W:
            W = W * ((V_X) @ H.T) / ((V * (W @ H - shift)) @ H.T)
            W = np.nan_to_num(W, nan=0, posinf=0)
    return H, W

# TODO: think of a better name for this algorithm.
class NMF:
    def __init__(self, X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike = None,
                 W_start: npt.ArrayLike = None, n_templates: int = 2, n_iter: int = 500):
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
            self.H = rng.uniform(0, 2, H_shape)

        if W_start is not None:
            self.W = np.asarray(W_start)
        else:
            self.W = rng.uniform(0, 2, W_shape)

        self.n_iter = n_iter

    def fit(self):
        """Fit this NMF object to noisy, possibly negative, data with weights.
        """
        self.H, self.W = fit_NMF(self.X, self.V, self.H, self.W, n_iter=self.n_iter)

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
        return fit_NMF(X, V, self.H, self.W, n_iter=self.n_iter, update_W=False)[0]


