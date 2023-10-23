#!/usr/bin/env python3

import fitsio
import numpy as np
import time

rng = np.random.default_rng(100921)

with fitsio.FITS("qsos_noisy.fits", "r") as h:
# Only load the first 10k spectra since we use random subsets for every test
# and this helps prevent memory overload for spectra we don't even use
# most of the time
    X_10k = h["FLUX"][:, :10000]
    V_10k = h["IVAR"][:, :10000]
    w_common = h["WAVELENGTH"].read()
    
idcs = np.arange(X_10k.shape[-1])

nan_eps = 1e-6
def split_pos_neg(A):
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

# The way this is set up this will return something of shape
#(num qso tests, num template tests, num dimension tests)
# template_grid, n_qsos_grid, n_dim_grid = np.meshgrid(template_tests, n_qsos_tests, n_dim_tests)

def resample(X, V, factor):
    # Sums bins [0:factor], [factor:factor*2] etc
    X_out = np.add.reduceat(X * V, np.arange(0, X.shape[0], factor), axis=0)
    V_out = np.add.reduceat(V, np.arange(0, X.shape[0], factor), axis=0)

    nz = (V_out != 0)
    X_out[nz] = X_out[nz] / V_out[nz]
    return X_out, V_out

n_runs = 100
n_hundreds = 5

print("Beginning parameter sweep...")

print("Starting QSO test")
n_qsos_tests = np.geomspace(50, 2000, 5, dtype=int)#(np.arange(40) + 1) * 50
qso_times = np.zeros((len(n_qsos_tests), n_runs))
# Use a fixed 5 templates for testing the qso tests
n_templates = 5

for i, n_qsos in enumerate(n_qsos_tests):
    test_idcs = rng.choice(idcs, size=n_qsos, replace=False)
    X = X_10k[:, test_idcs]
    V = V_10k[:, test_idcs]
    
    # Downsample once so reference is in the middle
    X, V = resample(X, V, 2)
    V_X = V * X   

    H_shape = (n_templates, X.shape[1])
    W_shape = (X.shape[0], n_templates)

    t_run = np.zeros(n_runs)
    # We run this test n_runs number of times with different starting points
    for a in range(n_runs):
        # Just copied the whole method in here to avoid timing any overhead in the functions
        H = rng.uniform(0, 1, H_shape)
        W = rng.uniform(0, 1, W_shape)

        t1 = time.time()

        # H-step
        W_VX = W.T @ V_X
        W_VX_pos, W_VX_neg = split_pos_neg(W_VX)

        H = H * (W_VX_pos) / (W.T @ (V * (W @ H)) + W_VX_neg)
        H = np.nan_to_num(H, nan=nan_eps, posinf=nan_eps)

        # W-step
        V_XH = V_X @ H.T
        V_XH_pos, V_XH_neg = split_pos_neg(V_XH)

        W = W * (V_XH_pos) / ((V * (W @ H)) @ H.T + V_XH_neg)
        W = np.nan_to_num(W, nan=nan_eps, posinf=nan_eps)

        t2 = time.time()
        t_run[a] = (t2 - t1)

    qso_times[i] = t_run


print("Finished QSO test")
np.save("qso_times.npy", qso_times)

print("Starting template test")
template_tests = np.arange(10) + 1
template_times = np.zeros((len(template_tests), n_runs))
# Use a fixed 500 qsos for testing the template tests
n_qsos = 750
test_idcs = rng.choice(idcs, size=n_qsos, replace=False)
X = X_10k[:, test_idcs]
V = V_10k[:, test_idcs]

# Downsample so reference is in the middle
X, V = resample(X, V, 2)
V_X = V * X

for i, n_templates in enumerate(template_tests):
    H_shape = (n_templates, X.shape[1])
    W_shape = (X.shape[0], n_templates)

    t_run = np.zeros(n_runs)
    V_X = X * V
    # We run this test n_runs number of times with different starting points
    for a in range(n_runs):
        H = rng.uniform(0, 1, H_shape)
        W = rng.uniform(0, 1, W_shape)

        t1 = time.time()

        # H-step
        W_VX = W.T @ V_X
        W_VX_pos, W_VX_neg = split_pos_neg(W_VX)

        H = H * (W_VX_pos) / (W.T @ (V * (W @ H)) + W_VX_neg)
        H = np.nan_to_num(H, nan=nan_eps, posinf=nan_eps)

        # W-step
        V_XH = V_X @ H.T
        V_XH_pos, V_XH_neg = split_pos_neg(V_XH)

        W = W * (V_XH_pos) / ((V * (W @ H)) @ H.T + V_XH_neg)
        W = np.nan_to_num(W, nan=nan_eps, posinf=nan_eps)

        t2 = time.time()
        t_run[a] = (t2 - t1)

    template_times[i] = t_run

print("Finished template test")
np.save("template_times.npy", template_times)
            
            
print("Starting dimension test")
n_dim_tests = np.asarray([1,2,4,8,16])#np.arange(4) + 1
dim_times = np.zeros((len(n_dim_tests), n_runs))

# Fixed qsos and 5 templates for the dimensionality test
n_qsos = 750
n_templates = 5
test_idcs = rng.choice(idcs, size=n_qsos, replace=False)
X_global = X_10k[:, test_idcs]
V_global = V_10k[:, test_idcs]

for i, ds_factor in enumerate(n_dim_tests):
    X, V = resample(X_global, V_global, ds_factor)
    V_X = V * X
    H_shape = (n_templates, X.shape[1])
    W_shape = (X.shape[0], n_templates)

    t_run = np.zeros(n_runs)
    # We run this test n_runs number of times with different starting points
    for a in range(n_runs):
        H = rng.uniform(0, 1, H_shape)
        W = rng.uniform(0, 1, W_shape)

        # Just copied the whole method in here to avoid timing any overhead
        t1 = time.time()

        # H-step
        W_VX = W.T @ V_X
        W_VX_pos, W_VX_neg = split_pos_neg(W_VX)

        H = H * (W_VX_pos) / (W.T @ (V * (W @ H)) + W_VX_neg)
        H = np.nan_to_num(H, nan=nan_eps, posinf=nan_eps)

        # W-step
        V_XH = V_X @ H.T
        V_XH_pos, V_XH_neg = split_pos_neg(V_XH)

        W = W * (V_XH_pos) / ((V * (W @ H)) @ H.T + V_XH_neg)
        W = np.nan_to_num(W, nan=nan_eps, posinf=nan_eps)

        t2 = time.time()
        t_run[a] = (t2 - t1)

    dim_times[i] = t_run
    
print("Finished dimension test")
np.save("dim_times.npy", dim_times)
        
            
