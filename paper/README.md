This folder contains jupyter notebooks and scripts necessary to reproduce the plots in the paper. The following index describes which notebook is linked to which plot:

1. `01-toy.ipynb` - This notebook contains the entirety of the toy problem, and the resultant plots, namely Figs 1-3. It also includes all the code to generate the dataset and the arbitrary shift Shift-NMF.
2. `02-spectra_plots.ipynb` - Fig 4. Loads a portion of the dataset, and selects three spectra to plot. 
3. `03-template_plots.ipynb` - Fig 5. Loads the templates, and plots them, along with the sum of the weights per pixel. 
4. `04-chi_2.ipynb` - Fig 6. Loads $\chi^2$ data (from chi_2.py), and makes a histogram of their differences.
5. `05-timing_plots.ipynb` - Fig 7. Loads the saved timing (from timing.py) to generate the plots. Includes both a horizontal and vertical version of the three panels joined as well as each individual panel.
6. `06-recon.ipynb` - Fig 8. Loads the dataset and splits out the validation set, then plots reconstructions for both the noisy and noise-free templates.

The following is the process to generate the data and train templates used in the above plots:

1. Generate 200k simqso qsos: `python run_simqso --seed 100921 --n_qsos 200000 --n_proc 128 -o qsos.fits`
2. Generate the noisy qsos and their corresponding rebinned and rescaled noise-free truths: `python add_noise.py --seed 100921 -o qsos -i qsos.fits --renorm`
3. Train the templates: `python train_templates.py --seed 100921 -i qsos_noisy.fits -o noisy --validate --algorithm both` to train both Nearly and Shift-NMF on the noiseless dataset, saving with the suffix "noisy", and including fitting the templates to the validation dataset. Repeat the same commend with `noiseless` in place of `noisy` to fit to the noise-free truth dataset.
4. Use the NNLS solver to fit the trained templates to the validation dataset with `python chi_2.py -i qsos_noisy.fits`.