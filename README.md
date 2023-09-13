# nmf_with_negative_data

This is a repository with a code snippet implementing two different NMF variants that allow and account for negative data with weights:

- Shift-NMF
- Nearly-NMF

For more details on how the algorithms work, check the methodology file.

The code is well documented, with extensive docstrings. The code provides both a direct interface (via a `fit_NMF` function) as well as an object-oriented approach (through an `NMF` object) that is designed to be familiar (but not identical) to those working with scikit-learn models.