import unittest

import numpy as np
from negative_noise_nmf import nmf

class TestInits(unittest.TestCase):
    def setUp(self):
        # This test tests the automatic initialization of the nmf package.

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


    def test_n_templates_match(self):
        V = np.ones_like(self.X)
        W_break = np.ones((self.X.shape[0], self.H_start.shape[0] + 1))

        with self.assertRaises(AssertionError):
            _ = nmf.fit_NMF(self.X, V, self.H_start, W_break)
            _ = nmf.NMF(self.X, V, self.H_start, W_break)

    def test_at_least_one_update_true(self):
        V = np.ones_like(self.X)

        with self.assertRaises(AssertionError):
            _ = nmf.fit_NMF(self.X, V, self.H_start, self.W_start, update_H=False, update_W=False)
            _ = nmf.NMF(self.X, V, self.H_start, self.W_start, update_H=False, update_W=False)

if __name__ == '__main__':
    unittest.main()