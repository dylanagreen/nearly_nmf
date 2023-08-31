import numpy as np
import numpy.typing as npt

def fit_NMF(X: npt.ArrayLike, V: npt.ArrayLike, H_start: npt.ArrayLike = None, 
            W_start: npt.ArrayLike = None, n_templates: int = 2, 
            n_iter: int = 500) ->  tuple[npt.NDArray, npt.NDArray]:
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

    Returns
    -------
    H : numpy.ndarray
        The fitted NMF coefficients with shape (n_templates, 
        n_observations). Each column represents the coefficients 
        corresponding to a colum of X.
    W : numpy.ndarray
        The fitted NMF templates with shape (n_dimensions, n_templates). 
        Each column represents one template.
    """
    X, V = np.asarray(X), np.asarray(V)
    
    if (H_start is not None) and (W_start is not None): 
        assert H_start.shape[0] == W_start.shape[1], "Number of templates does not match between H and W"
    if H_start is not None: n_templates = H_start.shape[0]
    if W_start is not None: n_templates = W_start.shape[1]
    
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
    
    update_H, update_W = True, True
    
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