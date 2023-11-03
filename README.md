# Nearly-NMF

This is a repository with package implementing two different NMF variants that allow and account for negative data with weights:

- Shift-NMF
- Nearly-NMF

The algorithms are doocumented in a paper (in prep.). Code related to plots and data in the paper are contained in their own repository at https://github.com/dylanagreen/nearly_nmf_paper.

The code is well documented, with extensive docstrings. The code provides both a direct interface (via a `fit_NMF` function) as well as an object-oriented approach (through an `NMF` object) that is designed to be familiar (but not identical) to those working with scikit-learn models.

The code is also optimized for gpu use, and cupy is a necessary dependency for this to work. Enable GPU running by passing the data arrays as cupy arrays rather than numpy arrays.

## Install
`nearly_nmf` can be installed with `pip install .`, which pulls in the necessary numpy dependency but **will not** install cupy, if you want to enable gpu use you must install cupy yourself. 
