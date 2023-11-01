# Nearly-NMF

This is a repository with a code snippet implementing two different NMF variants that allow and account for negative data with weights:

- Shift-NMF
- Nearly-NMF

The algorithms are doocumented in a paper (in prep.).

The code is well documented, with extensive docstrings. The code provides both a direct interface (via a `fit_NMF` function) as well as an object-oriented approach (through an `NMF` object) that is designed to be familiar (but not identical) to those working with scikit-learn models.

The code is also optimized for gpu use, and cupy is a necessary dependency for this to work. Enable GPU running by passing the data arrays as cupy arrays rather than numpy arrays.