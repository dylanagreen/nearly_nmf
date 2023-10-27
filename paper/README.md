This folder contains jupyter notebooks and scripts necessary to reproduce the plots in the paper. The following index describes which notebook is linked to which plot:

1. `01-toy.ipynb` - This notebook contains the entirety of the toy problem, and the resultant plots, namely Figs 1-3. It also includes all the code to generate the dataset and the arbitrary shift Shift-NMF.
2. `02-spectra_plots.ipynb` - Fig 4. Loads a portion of the dataset, and selects three spectra to plot. 
3. `03-template_plots.ipynb` - Fig 5. Loads the templates, and plots them, along with the sum of the weights per pixel. 
4. `04-chi_2.ipynb` - Fig 6. Loads $\chi^2$ data (from chi_2.py), and makes a histogram of their differences.
5. `05-timing_plots.ipynb` - Fig 7. Loads the saved timing (from timing.py) to generate the plots. Includes both a horizontal and vertical version of the three panels joined as well as each individual panel.
6. `06-recon.ipynb` - Fig 8. Loads the dataset and splits out the validation set, then plots reconstructions for both the noisy and noise-free templates.