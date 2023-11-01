#!/usr/bin/env python3
import fitsio
import numpy as np

import time
import argparse
p = argparse.ArgumentParser()
p.add_argument('-i', '--in_file', type=str, help="Input filename")
p.add_argument('-o', '--out_suffix', type=str, help="Output filename suffix")

args = p.parse_args()

n_qsos = 130000
with fitsio.FITS(args.in_file, "r") as h:
    X = h["FLUX"][:, :n_qsos]
    V = h["IVAR"][:, :n_qsos]
    w_grid = h["WAVELENGTH"].read()
    z = h["Z"].read()

weights_sum = np.sum(V, axis=1)
avg_spec = np.sum(X * V, axis=1) / weights_sum

np.save(f"weights_sum_{args.out_suffix}.npy", weights_sum)
np.save(f"avg_spec_{args.out_suffix}.npy", avg_spec)
    