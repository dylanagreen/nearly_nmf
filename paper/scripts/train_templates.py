#!/usr/bin/env python3
import fitsio
import numpy as np
import cupy as cp

import sys
sys.path.insert(0, "../../py/")
from negative_noise_nmf import nmf

import time
import argparse
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=100921, help="Random seed.")
p.add_argument('-i', '--in_file', type=str, help="Input filename.")
p.add_argument('-o', '--out_suffix', type=str, help="Output template name suffix.")
p.add_argument('--n_templates', type=int, default=5, help="Number of templates to train.")
p.add_argument('--validate', action="store_true", help="Whether to fit the validation set or not.")
p.add_argument('-a', '--algorithm', choices=["shift", "nearly", "both"], help="Which algorithm to train")

args = p.parse_args()

n_qsos = 130000
with fitsio.FITS(args.in_file, "r") as h:
    X = h["FLUX"][:, :n_qsos]
    V = h["IVAR"][:, :n_qsos]
    w_grid = h["WAVELENGTH"].read()
    z = h["Z"].read()
    
rng = np.random.default_rng(args.seed)

train_shift = (args.algorithm == "both") | (args.algorithm == "shift")
train_nearly = (args.algorithm == "both") | (args.algorithm == "nearly")

print("Train shift?", train_shift)
print("Train nearly?", train_nearly)

M = V != 0
n_cover = np.sum(M, axis=1)

# Cut pixels off the bottom to avoid edge effects
cut_lower = 50
# Find the upper index at which the same number of spectra are in the last
# pixel as the new (post cut) first pixel
n_spec_lower = n_cover[cut_lower]
cut_upper = np.argmax(n_cover[500:] < n_spec_lower) + 500

# Round the upper limit so that we have a nice round number
# of pixels in the output, rounding to the nearest 50
diff = cut_upper - cut_lower
diff = int(np.round(diff / 50) * 50)
cut_upper = cut_lower + diff

keep_range = np.s_[cut_lower:cut_upper]

print("Keep range:", cut_lower, cut_upper)
print(diff)

X = cp.asarray(X[keep_range, :])
V = cp.asarray(V[keep_range, :])

print(X.shape)

# Helpful to abstract these shapes for later
H_shape = (args.n_templates, X.shape[1])
W_shape = (X.shape[0], args.n_templates)

# Sequential start so we can use a smooth flat W start
# for every template
H_start = rng.uniform(0, 1, H_shape)
W_start = np.ones(W_shape) * np.arange(W_shape[0])[::-1][:, None] / W_shape[0] + 0.05

if train_nearly:
    H_nearly = cp.array(H_start, copy=True)
    W_nearly = cp.array(W_start, copy=True)

if train_shift:
    H_shift = cp.array(H_start, copy=True)
    W_shift = cp.array(W_start, copy=True)


print("Starting iteration")
for i in range(args.n_templates):
    print("iteration", i + 1, H_shift[:(i + 1), :].shape)
    if train_nearly:
        t1 = time.time()
        # Earlier templates do not get fixed so will get the cumulative amount of
        # iterations to train, so we can speed this up by only templates for
        # a  less amount of iterations to get "close" to the final before 
        # doing the full number for the final templates
        n_iter = 50
        H_itr, W_itr = nmf.nearly_NMF(X, V, H_nearly[:(i + 1), :], W_nearly[:, :(i + 1)], n_iter=n_iter)
        t2 = time.time()
        print(t2 - t1)
        # Place the template we train into the array for the next iteration
        H_nearly[:(i + 1), :] = H_itr
        W_nearly[:, :(i + 1)] = W_itr

    if train_shift:
        t1 = time.time()
        # larger number of iterations for shift NMF due to convergence speed
        n_iter *= 2
        H_itr, W_itr = nmf.shift_NMF(X, V, H_shift[:(i + 1), :], W_shift[:, :(i + 1)], n_iter=n_iter)
        t2 = time.time()
        print(t2 - t1)

        # Place the template we train into the array for the next iteration
        H_shift[:(i + 1), :] = H_itr
        W_shift[:, :(i + 1)] = W_itr

# Refining the templates
t1 = time.time()
if train_nearly: H_nearly, W_nearly = nmf.nearly_NMF(X, V, H_nearly, W_nearly, n_iter=1000)
t2 = time.time()
if train_shift: H_shift, W_shift = nmf.shift_NMF(X, V, H_shift, W_shift, n_iter=2000)
t3 = time.time()
print(t2 - t1, t3 - t2)

print("Saving templates")
cp.save(f"templates_no_scale/W_nearly_{args.out_suffix}.npy", W_nearly)
cp.save(f"templates_no_scale/H_nearly_{args.out_suffix}.npy", H_nearly)

cp.save(f"templates_no_scale/W_shift_{args.out_suffix}.npy", W_shift)
cp.save(f"templates_no_scale/H_shift_{args.out_suffix}.npy", H_shift)

if args.validate:
    with fitsio.FITS(args.in_file, "r") as h:
        X = h["FLUX"][:, n_qsos:]
        V = h["IVAR"][:, n_qsos:]
        w_grid = h["WAVELENGTH"].read()
        z = h["Z"].read()
        
    X = cp.asarray(X[keep_range, :])
    V = cp.asarray(V[keep_range, :])
    
    # Reinitialize these
    # Helpful to abstract these shapes for later
    H_shape = (args.n_templates, X.shape[1])
    H_start = rng.uniform(0, 1, H_shape)
    H_nearly = cp.array(H_start, copy=True)
    H_shift = cp.array(H_start, copy=True)
    
    # Refining the templates
    t1 = time.time()
    H_nearly, _ = nmf.nearly_NMF(X, V, H_nearly, W_nearly, n_iter=100, update_W=False)
    t2 = time.time()
    H_shift, _ = nmf.shift_NMF(X, V, H_shift, W_shift, n_iter=100, update_W=False)
    t3 = time.time()
    print(t2 - t1, t3 - t2)

    print("Saving validation coefficients")
    cp.save(f"templates_no_scale/H_nearly_{args.out_suffix}_validate.npy", H_nearly)
    cp.save(f"templates_no_scale/H_shift_{args.out_suffix}_validate.npy", H_shift)

