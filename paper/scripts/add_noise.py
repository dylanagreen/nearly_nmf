#!/usr/bin/env python3

"""
Add noise to the quasar spectra

To generate the renormalized, noisy, dataset:
python add_noise --seed 100921 -o qsos_noisy.fits --renorm --add_noise
"""

import ast
import time

import numpy as np
import matplotlib.pyplot as plt
import fitsio
from astropy.table import Table

# These are necessary to generate the scaling quasar
from astropy.cosmology import Planck13
from simqso.sqgrids import *
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk,buildQsoSpectrum
from simqso.sqmodels import BOSS_DR9_PLEpivot,get_BossDr9_model_vars

# Used in rebinning
from scipy.ndimage import binary_erosion

def get_wave(wavemin=3600, wavemax=10000, dloglam=1e-4):
    """
    Return logarithmic wavelength array from wavemin to wavemax step dloglam

    Args:
        wavemin: minimum wavelength
        wavemax: maximum wavelength
        dloglam: stepsize in log(wave)

    Return: wave array
    """
    n = np.log10(wavemax/wavemin) / dloglam
    wave = 10**(np.log10(wavemin) + dloglam*np.arange(n))
    return wave

def deshift_and_stack(fl, redshifts, w_obs):
    X = []
    waves = []
        
    for i in range(fl.shape[0]):
        z = redshifts[i]
        shifted_wave = w_obs / (1 + z)
        
        ergs_to_photons = shifted_wave / np.gradient(shifted_wave)
        X.append(fl[i] * ergs_to_photons)
        waves.append(shifted_wave)
        
    return X, waves


def generate_scaling_qso(wave, seed=100921, z_gen=2.1):
    # Generates a single QSO with the same parameters but covering
    # the entire covered wavelength range for normalization purposes
    kcorr = sqbase.ContinuumKCorr('DECam-r',1450,effWaveBand='SDSS-r')
    qsos = generateQlfPoints(BOSS_DR9_PLEpivot(cosmo=Planck13),
                             (16, 16), (2, 3),
                             kcorr=kcorr, zin=[z_gen],
                             qlfseed=seed, gridseed=seed)

    sedVars = get_BossDr9_model_vars(qsos, wave, noforest=True)
    qsos.addVars(sedVars)
    qsos.loadPhotoMap([('DECam','DECaLS'),('WISE','AllWISE')])

    _, spectra_for_norm = buildSpectraBulk(wave * (1 + z_gen), qsos, saveSpectra=True, maxIter=5, verbose=10)
    spectra_for_norm = spectra_for_norm[0]
    # Normalize to "approximately median of 1 in erg space"
    # Since we'll do the normalization to the template in erg space we don't need
    # to convert to photon space
    return spectra_for_norm / np.median(spectra_for_norm)

def rebin_to_common(w_in, w_out, fl, iv, fl_idx):
    # This spectra wasn't contained in cutout area
    if len(w_in) < 1:
        return None, None
    
    l_min = w_out[0]
    l_max = w_out[-1]
    dl = w_out[1] - w_out[0]
    
    # We rebin by considering how much (percent) of flux
    # should go to the lower bin and how much to the higher bin
    # Since both grids are on the exact same spacing
    # this is just a matter of finding how much the two
    # bins overlap each other by.
    # Pick the midpoint of the input array which based on how
    # we set up the output array should always be within the grid
    # and not near the output edges
    # Then find the two bin edges on either side that contain this midpoint.
    base = w_in[len(w_in) // 2]
    idx_above = np.argmax(w_out > base)
    idx_below = idx_above - 1
    
    assert w_out[idx_below] < base
    d_high = w_out[idx_above] - base
    d_low = base - w_out[idx_below]
    
    percent_low = d_low / dl
    percent_high = d_high / dl

    # Inverting the inverse variance to get the variance
    var = iv
    nz = iv != 0
    var[nz] = 1 / var[nz]

    # Gets the bin number in the output grid that includes the lower edge 
    # of the bins in the input grid. 
    # Can add 1 for upper edge since both grids have the same spacing.
    bin_low = np.floor((w_in - l_min) / dl).astype(int)
    good_low = (bin_low >= 0) & (bin_low < nbins_final)
    bin_high = bin_low + 1
    good_high = (bin_high >= 0) & (bin_high < nbins_final)
        
    c_low = np.bincount(bin_low[good_low], weights=fl[good_low] * percent_low, minlength=len(w_out))
    c_high = np.bincount(bin_high[good_high], weights=fl[good_high] * percent_high, minlength=len(w_out))
    
    var_low = np.bincount(bin_low[good_low], weights=var[good_low] * percent_low ** 2, minlength=len(w_out))
    var_high = np.bincount(bin_high[good_high], weights=var[good_high] * percent_high ** 2, minlength=len(w_out))
    
    fl_out = c_low + c_high
    # For masking out pixels that aren't in the grid
    mask = np.zeros_like(w_out)
    mask[fl_out != 0] = 1
    
    # The binary erosion will just expand the zeros by 
    # one eliminating the bordering 1. This avoids an off by
    # one error in the mask compared to which pixels actually
    # have data
    mask = binary_erosion(mask, border_value=1)
    
    # Inverting the variance back to inverse variance
    var_out = var_low + var_high
    iv_out = var_out
    # nz = np.abs(var_out) > 1e-2 # Mitigating small factor errors
    nz = var_out != 0
    iv_out[nz] = 1 / var_out[nz]

    return fl_out * mask, iv_out * mask, mask


def add_noise(fl_shift, w_shift, seed):
    # TODO Vectorize this?
    rng = np.random.default_rng(seed)
    fl_noisy = []
    iv_noisy = []
    
    fl_noise_free = []
    iv_noise_free = []
    # gaussian = add_noise
    
    for i in range(len(fl_shift)):
        ergs_to_photons = w_shift[i] / np.gradient(w_shift[i])
        
        num_photons = rng.uniform(10, 40)
        fl_scale = fl_shift[i] * num_photons / np.median(fl_shift[i])

        x_poisson = rng.poisson(fl_scale.clip(min=0))

            
        # Estimating the variance from the poisson sim
        x_poisson = x_poisson / ergs_to_photons
        var_poisson = fl_scale / (ergs_to_photons ** 2)
        
        # Pick an SNR between 1 and 2 so the mean SNR is vaguely 1.5
        # This corresponds roughly with 10% negativity.
        # Basing signal on the noise-free truth for accuracy
        fl_true = fl_scale / ergs_to_photons
        signal = np.mean(fl_true ** 2)
        snr_choice = rng.uniform(1, 2)
        noise_sigma = np.sqrt(signal) / snr_choice
        noise = rng.normal(0, noise_sigma, fl_shift[i].shape)
        var_noise = noise_sigma ** 2

        fl_sim = x_poisson + noise
        iv_noisy.append(1 / (var_poisson + var_noise))
        fl_noisy.append(fl_sim)
        
        fl_noise_free.append(fl_true)
        iv_noise_free.append((fl_scale != 0))
        
    return fl_noisy, iv_noisy, fl_noise_free, iv_noise_free


def rebin_all(fl, iv, waves, w_rest, w_scale, spectra_for_norm, renorm=True, has_noise=True):
    fl_rebinned = []
    iv_rebinned = []
    masks = []
    print(len(w_rest))
    for i in range(len(fl)):
        fl_1, iv_1, m = rebin_to_common(np.log10(waves[i]), np.log10(w_rest), fl[i], iv[i], i)
        if fl_1 is not None:
            if np.all(fl_1 < 0): print(i)
            
            if renorm:
                # Making sure to renorm the median of the spectra
                # to only the median of that region covered by
                # the global spectra
                norm_min = np.argmax(w_scale > waves[i][0])
                norm_max = np.argmax(w_scale > waves[i][-1])

                norm_val = np.median(spectra_for_norm[norm_min:norm_max])
                spec_val = np.median(fl_1[fl_1 != 0])

                scale = norm_val / np.abs(spec_val)
                normed = fl_1 * scale
                normed_iv = iv_1 / (scale ** 2)

            else:
                normed = fl_1
                normed_iv = iv_1
            
            fl_rebinned.append(normed)
            masks.append(m)
            if has_noise:
                iv_rebinned.append(normed_iv)
            else:
                iv_rebinned.append(m.astype(float))
                
    return fl_rebinned, iv_rebinned, masks
  
#-------------------------------------------------------------------------

import argparse
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=1234, help="Random seed")
p.add_argument('-o', '--out', type=str, help="Output filename")
p.add_argument('-i', '--in_file', type=str, help="Filename of generated simqso quasars")
p.add_argument('--renorm', action="store_true", help="Whether to renormalize or not.")
# p.add_argument('--add_noise', action="store_true", help="Whether to add noise or not.")

args = p.parse_args()

# print(args.add_noise)
print(args.renorm)

# All sorts of settable hyperparameters
seed = args.seed

z_min = 0
z_max = 4

# Full range to fit for qsos from z=0 to z=4
# that overlap the eboss grid
eboss_min, eboss_max = 3600.,10000

# Full range wave
w_scale = get_wave(eboss_min / (1 + z_max), eboss_max / (1 + z_min))
nbins = len(w_scale)

# Making the truncated grid a nice even number of pixels
nbins_final = 11400
trunc = (nbins - nbins_final) // 2
w_rest = w_scale[trunc:-trunc]
if len(w_rest) == 11401: w_rest = w_rest[:-1] # Odd number fix
l_min = np.log10(w_rest[0])
l_max = np.log10(w_rest[-1])

with fitsio.FITS(args.in_file) as h:
    fl = h["FLUX"].read()
    w_obs = h["WAVELENGTH"].read()
    redshifts = h["METADATA"].read("Z")

print("Deshifting spectra...")
fl_shift, w_shift = deshift_and_stack(fl, redshifts, w_obs)

print("Adding noise...")
fl_noisy, iv_noisy, fl_noise_free, iv_noise_free = add_noise(fl_shift, w_shift, args.seed)

if args.renorm:
    print("Generating scaling spectra...")  
    spectra_for_norm = generate_scaling_qso(w_scale)
else:
    spectra_for_norm = None
print("Rebinning spectra...")  
fl_rebinned, iv_rebinned, masks = rebin_all(fl_noisy, iv_noisy, w_shift, w_rest, w_scale, spectra_for_norm, renorm=args.renorm, has_noise=True)
X = np.vstack(fl_rebinned).T
V = np.vstack(iv_rebinned).T

fl_rebinned, iv_rebinned, masks = rebin_all(fl_noise_free, iv_noise_free, w_shift, w_rest, w_scale, spectra_for_norm, renorm=args.renorm, has_noise=False)
X_truth = np.vstack(fl_rebinned).T
V_truth = np.vstack(iv_rebinned).T

print("post tests")
print(np.where(~X.any(axis=0))[0])

# Just in case we have a divide by zero error
V = np.nan_to_num(V, nan=0, posinf=0)
X = np.nan_to_num(X, nan=0, posinf=0)
V_truth = np.nan_to_num(V_truth, nan=0, posinf=0)
X_truth = np.nan_to_num(X_truth, nan=0, posinf=0)

print(np.where(~X.any(axis=0))[0])

print("Negative fraction:", np.sum(X < 0) / (X[V != 0].size))
print("Missing fraction:", 1 - (np.sum(V != 0) / V.size))


save_name = f"{args.out}_noisy.fits"
print(f"Saving {save_name}...")
if os.path.isfile(save_name): os.remove(save_name)
with fitsio.FITS(save_name, "rw") as h:
    h.write(X, extname="FLUX")
    h.write(V, extname="IVAR")
    h.write(w_rest, extname="WAVELENGTH")
    h.write(redshifts, extname="Z")
    
save_name = f"{args.out}_noiseless.fits"
print(f"Saving {save_name}...")
if os.path.isfile(save_name): os.remove(save_name)
with fitsio.FITS(save_name, "rw") as h:
    h.write(X_truth, extname="FLUX")
    h.write(V_truth, extname="IVAR")
    h.write(w_rest, extname="WAVELENGTH")
    h.write(redshifts, extname="Z")