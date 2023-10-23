#!/usr/bin/env python3
import fitsio
import numpy as np
import cupy as cp

from negative_noise_nmf import nmf

import time
import argparse
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=100921, help="Random seed")
p.add_argument('-i', '--in_file', type=str, help="Input filename")

args = p.parse_args()

n_qsos = 130000
with fitsio.FITS(args.in_file, "r") as h:
    X = h["FLUX"][:, :n_qsos]
    V = h["IVAR"][:, :n_qsos]
    w_grid = h["WAVELENGTH"].read()
    z = h["Z"].read()
    
rng = np.random.default_rng(args.seed)

print(X.shape)

# Abstracting this variable
n_templates = 5

# Helpful to abstract these shapes for later
H_shape = (n_templates, X.shape[1])
W_shape = (X.shape[0], n_templates)

# Sequential start so we can use a smooth flat W start
# for every template
H_start = rng.uniform(0, 1, H_shape)
W_start = np.ones(W_shape)

# Sequentially construct the n_templates templates
H_nearly = cp.array(H_start, copy=True)
W_nearly = cp.array(W_start, copy=True)

H_shift = cp.array(H_start, copy=True)
W_shift = cp.array(W_start, copy=True)

X = cp.asarray(X)
V = cp.asarray(V)

print("Starting iteration")
for i in range(n_templates):
    print("iteration", i + 1, H_shift[:(i + 1), :].shape)
    t1 = time.time()
    # Earlier templates do not get fixed so will get the cumulative amount of
    # iterations to train, so we can speed this up by only templates for
    # a  less amount of iterations to get "close" to the final before 
    # doing the full number for the final templates
    n_iter = 20
    H_itr, W_itr = nmf.nearly_NMF(X, V, H_nearly[:(i + 1), :], W_nearly[:, :(i + 1)], n_iter=n_iter)
    t2 = time.time()
    print(t2 - t1)
    # Place the template we train into the array for the next iteration
    H_nearly[:(i + 1), :] = H_itr
    W_nearly[:, :(i + 1)] = W_itr

    t1 = time.time()
    # Double the iterations for shift NMF due to convergence speed
    n_iter = 40
    H_itr, W_itr = nmf.shift_NMF(X, V, H_shift[:(i + 1), :], W_shift[:, :(i + 1)], n_iter=n_iter)
    t2 = time.time()
    print(t2 - t1)
    
    # Place the template we train into the array for the next iteration
    H_shift[:(i + 1), :] = H_itr
    W_shift[:, :(i + 1)] = W_itr

# Refining the templates
t1 = time.time()
H_nearly, W_nearly = nmf.nearly_NMF(X, V, H_nearly, W_nearly, n_iter=500)
t2 = time.time()
H_shift, W_shift = nmf.shift_NMF(X, V, H_shift, W_shift, n_iter=1000)
t3 = time.time()
print(t2 - t1, t3 - t2)

print("Saving templates")
cp.save("templates/W_nearly_no_lya.npy", W_nearly)
cp.save("templates/H_nearly_no_lya.npy", H_nearly)

cp.save("templates/W_shift_no_lya.npy", W_shift)
cp.save("templates/H_shift_no_lya.npy", H_shift)
