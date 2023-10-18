import unittest
import pathlib

import numpy as np
from negative_noise_nmf import nmf

file_loc = pathlib.Path(__file__).parent.resolve() / "test_data"

class TestNoisy(unittest.TestCase):
    """
    This test admits data that is negative, and the goal here is not to check
    that Nearly-NMF and Shift-NMF return the same value as each other, but rather
    that they return the same value as when I finalized the algorithms to ensure
    that later code base changes don't change the output of the fits. Ideally
    any code changes shouldn't change the underlying algorithm, which as of when
    this test was written matches exactly the mathematics.

    The dataset is 500 exposures of a double gaussian, the same dataset used
    in the paper.
    """
    def setUp(self):
        self.rng = np.random.default_rng(100921)

        self.X = np.load(file_loc / "doublet_dataset.npy")
        self.V = np.load(file_loc / "doublet_weights.npy")

        self.H_nearly_truth = np.load(file_loc / "H_nearly_truth.npy")
        self.W_nearly_truth = np.load(file_loc / "W_nearly_truth.npy")
        self.H_shift_truth = np.load(file_loc / "H_shift_truth.npy")
        self.W_shift_truth = np.load(file_loc / "W_shift_truth.npy")

        # How mant iterations I used to generate the truth
        self.n_iter = 300

        # Random coefficients, smooth starts (the same as I used to generate
        # the ground truth)
        self.H_start = self.rng.uniform(size=self.H_nearly_truth.shape)
        self.W_start = np.ones(self.W_nearly_truth.shape)

    def test_object_oriented_nearly(self):
        # Test both the object oriented front end and helper function to
        # ensure we don't introduce differences in those functions.
        obj_nearly = nmf.NMF(self.X, self.V, self.H_start, self.W_start,
                         n_iter=self.n_iter, algorithm="nearly", return_chi_2=False)
        obj_nearly.fit()

        self.assertTrue(np.allclose(obj_nearly.H, self.H_nearly_truth))
        self.assertTrue(np.allclose(obj_nearly.W, self.W_nearly_truth))

    def test_object_oriented_shift(self):
        # Test both the object oriented front end and helper function to
        # ensure we don't introduce differences in those functions.
        obj_shift = nmf.NMF(self.X, self.V, self.H_start, self.W_start,
                         n_iter=self.n_iter, algorithm="shift", return_chi_2=False)
        obj_shift.fit()

        self.assertTrue(np.allclose(obj_shift.H, self.H_shift_truth))
        self.assertTrue(np.allclose(obj_shift.W, self.W_shift_truth))

    def test_helper_function_nearly(self):
        # Test both the object oriented front end and helper function to
        # ensure we don't introduce differences in those functions.
        H_nearly, W_nearly = nmf.fit_NMF(self.X, self.V,
                                                       self.H_start, self.W_start,
                                                       n_iter=self.n_iter,
                                                       return_chi_2=False,
                                                       algorithm="nearly")

        self.assertTrue(np.allclose(H_nearly, self.H_nearly_truth))
        self.assertTrue(np.allclose(W_nearly, self.W_nearly_truth))

    def test_helper_function_shift(self):
        # Test both the object oriented front end and helper function to
        # ensure we don't introduce differences in those functions.
        H_shift, W_shift = nmf.fit_NMF(self.X, self.V,
                                                       self.H_start, self.W_start,
                                                       n_iter=self.n_iter,
                                                       return_chi_2=False,
                                                       algorithm="shift")

        self.assertTrue(np.allclose(H_shift, self.H_shift_truth))
        self.assertTrue(np.allclose(W_shift, self.W_shift_truth))


if __name__ == '__main__':
    unittest.main()