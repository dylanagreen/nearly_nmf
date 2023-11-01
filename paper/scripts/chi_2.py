#!/usr/bin/env python3
import fitsio
import numpy as np
from scipy.optimize import nnls

import time
import argparse
p = argparse.ArgumentParser()
p.add_argument('-i', '--in_file', type=str, help="Input filename")

args = p.parse_args()

n_qsos = 130000
with fitsio.FITS(args.in_file, "r") as h:
    X = h["FLUX"][:, n_qsos:]
    V = h["IVAR"][:, n_qsos:]

# This from the output of the training script but could be
# found by loading the data and replicating the same process
# from that script
keep_range = np.s_[50:11100]
    
V_X = V * X
W_noisy = np.load("templates/W_nearly_noisy.npy")
W_noiseless  = np.load("templates/W_nearly_noiseless.npy")

print("Fitting noise-free templates")
H_noiseless_nnls = np.zeros((W_noiseless.shape[-1], V_X.shape[-1]))
chi_2_noiseless_nnls = []
for i in range(V_X.shape[-1]):
    W = V[keep_range, i][:, None] * W_noiseless
    H_0, _ = nnls(W, V_X[keep_range, i])
    H_noiseless_nnls[:, i] = H_0

print("Fitting noisy templates")
H_noisy_nnls = np.zeros((W_noisy.shape[-1], V_X.shape[-1]))
chi_2_noisy_nnls = []
for i in range(V_X.shape[-1]):
    W = V[keep_range, i][:, None] * W_noisy
    H_0, _ = nnls(W, V_X[keep_range, i])
    H_noisy_nnls[:, i] = H_0
    
recon_noisy_nnls = W_noisy @ H_noisy_nnls
recon_noiseless_nnls = W_noiseless @ H_noiseless_nnls

print("Getting chi^2")
chi_2_noisy_nnls = np.sum(V[keep_range,:] * (X[keep_range,:] - recon_noisy_nnls) ** 2, axis=0)
chi_2_noiseless_nnls = np.sum(V[keep_range,:] * (X[keep_range,:] - recon_noiseless_nnls) ** 2, axis=0)

print("Saving")
np.save("chi_2_noisy_nnls.npy", chi_2_noisy_nnls)
np.save("chi_2_noiseless_nnls.npy", chi_2_noiseless_nnls)
    