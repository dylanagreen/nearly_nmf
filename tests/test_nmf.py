import unittest

import numpy as np
from negative_noise_nmf import nmf

class TestNoiseFree(unittest.TestCase):
    def setUp(self):
         # This test is for data entirely non-negative. We expect that Nearly-NMF
        # and Shift-NMF should return identical results in this case. All
        # weights are equal and set to 1.

        self.rng = np.random.default_rng(100921)

        # Generate data on a grid from 0 to 1
        x = np.linspace(0, 1, 30)
        n_templates = 5

        # Setting up the first n_templates legendre polynomials, on the interval 0 to 1
        # instead of -1 to 1. I've also vertically shited these so that their
        # minimum value is 0, so they're all non-negative.
        W_true = np.ones((len(x), n_templates))
        W_true[:, 1] = 2 * x
        W_true[:, 2] = 6 * (x ** 2 - x) + 1.5
        W_true[:, 3] = 20 * x ** 3 - 30 * x ** 2 + 12 * x
        W_true[:, 4] = 70 * x ** 4 - 140 * x ** 3 + 90 * x ** 2 - 20 * x + 1.5

        # Generate the test dataset
        n_test = 500
        H_true = self.rng.uniform(0, 2, (W_true.shape[-1], n_test))
        self.X = W_true @ H_true

        # Random starts. The resultant templates will look noisy but we don't
        # care about the actul structure of the templates for this test.
        self.H_start = self.rng.uniform(size=H_true.shape)
        self.W_start = self.rng.uniform(size=W_true.shape)


    def test_nonnegative_equal_weights(self):
        V = np.ones_like(self.X)
        H_nearly, W_nearly, chi_2_nearly = nmf.nearly_NMF(self.X, V,
                                                          self.H_start, self.W_start,
                                                          n_iter=100,
                                                          return_chi_2=True)
        H_shift, W_shift, chi_2_shift = nmf.shift_NMF(self.X, V,
                                                      self.H_start, self.W_start,
                                                      n_iter=100,
                                                      return_chi_2=True)

        self.assertTrue(np.allclose(H_nearly, H_shift))
        self.assertTrue(np.allclose(W_nearly, W_shift))
        self.assertTrue(np.allclose(chi_2_shift, chi_2_nearly))

    def test_nonnegative_different_weights(self):
        # Completely random weights, they don't matter that much anyway
        # we just want to see that the methods are equivalent.
        V = self.rng.uniform(0, 2, self.X.shape)
        H_nearly, W_nearly, chi_2_nearly = nmf.nearly_NMF(self.X, V,
                                                          self.H_start, self.W_start,
                                                          n_iter=100,
                                                          return_chi_2=True)
        H_shift, W_shift, chi_2_shift = nmf.shift_NMF(self.X, V,
                                                      self.H_start, self.W_start,
                                                      n_iter=100,
                                                      return_chi_2=True)

        self.assertTrue(np.allclose(H_nearly, H_shift))
        self.assertTrue(np.allclose(W_nearly, W_shift))
        self.assertTrue(np.allclose(chi_2_shift, chi_2_nearly))

if __name__ == '__main__':
    unittest.main()