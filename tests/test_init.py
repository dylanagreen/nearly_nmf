import unittest

import numpy as np
from nearly_nmf import nmf

class TestAsserts(unittest.TestCase):
    def setUp(self):
        # This test tests the automatic initialization of the nmf package.
        self.rng = np.random.default_rng(100921)

        # Generate data on a grid from 0 to 1
        x = np.linspace(0, 1, 30)
        n_templates = 5

        # Setting up the first n_templates legendre polynomials, on the interval 0 to 1
        # instead of -1 to 1. I've also vertically shifted these so that their
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
        # care about the actual structure of the templates for this test.
        self.H_start = self.rng.uniform(size=H_true.shape)
        self.W_start = self.rng.uniform(size=W_true.shape)


    def test_n_templates_match(self):
        V = np.ones_like(self.X)
        W_break = np.ones((self.X.shape[0], self.H_start.shape[0] + 1))

        with self.assertRaises(AssertionError):
            _ = nmf.fit_NMF(self.X, V, self.H_start, W_break)

        # Separate block because the assertion will prevent it from running otherwise
        with self.assertRaises(AssertionError):
            _ = nmf.NMF(self.X, V, self.H_start, W_break)

    def test_at_least_one_update_true(self):
        V = np.ones_like(self.X)

        with self.assertRaises(AssertionError):
            _ = nmf.fit_NMF(self.X, V, self.H_start, self.W_start, update_H=False, update_W=False)

class TestInits(unittest.TestCase):
    def setUp(self):
        # This test tests the automatic initialization of the nmf package.
        self.rng = np.random.default_rng(100921)

        # Generate data on a grid from 0 to 1
        x = np.linspace(0, 1, 30)
        n_templates = 5

        # Setting up the first n_templates legendre polynomials, on the interval 0 to 1
        # instead of -1 to 1. I've also vertically shifted these so that their
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
        # care about the actual structure of the templates for this test.
        self.H_start = self.rng.uniform(size=H_true.shape)
        self.W_start = self.rng.uniform(size=W_true.shape)
        self.n_iter = 1 # Set this to 1 just to also make sure that none of the shapes explode


    def test_helper_func_H_init(self):
        V = np.ones_like(self.X)

        # Checks that H is initialized to the correct shape.
        # It should infer n_templates from W.
        H, W = nmf.fit_NMF(self.X, V, W_start=self.W_start, n_iter=self.n_iter)
        self.assertEqual(H.shape, self.H_start.shape)

        # Checks transposed
        H, W = nmf.fit_NMF(self.X.T, V.T, W_start=self.W_start.T, n_iter=self.n_iter, transpose=True)
        self.assertEqual(H.shape, self.H_start.T.shape)

    def test_helper_func_W_init(self):
        V = np.ones_like(self.X)

        # Checks that W is initialized to the correct shape
        # It should infer n_templates from H.
        H, W = nmf.fit_NMF(self.X, V, H_start=self.H_start, n_iter=self.n_iter)
        self.assertEqual(W.shape, self.W_start.shape)

        # Checks transposed
        H, W = nmf.fit_NMF(self.X.T, V.T, H_start=self.H_start.T, n_iter=self.n_iter, transpose=True)
        self.assertEqual(W.shape, self.W_start.T.shape)

    def test_object_oriented_H_init(self):
        V = np.ones_like(self.X)

        # Checks that H is initialized to the correct shape.
        # It should infer n_templates from W.
        obj = nmf.NMF(self.X, V, W_start=self.W_start,
                    n_iter=self.n_iter, algorithm="nearly")
        self.assertEqual(obj.H.shape, self.H_start.shape)

        # Check transpose
        obj = nmf.NMF(self.X.T, V.T, W_start=self.W_start.T,
                    n_iter=self.n_iter, algorithm="nearly", transpose=True)
        self.assertEqual(obj.H.shape, self.H_start.T.shape)

    def test_object_oriented_W_init(self):
        V = np.ones_like(self.X)

        # Checks that H is initialized to the correct shape.
        # It should infer n_templates from W.
        obj = nmf.NMF(self.X, V, H_start=self.H_start,
                    n_iter=self.n_iter, algorithm="nearly")
        self.assertEqual(obj.W.shape, self.W_start.shape)

        # Check transpose
        obj = nmf.NMF(self.X.T, V.T, W_start=self.W_start.T,
                    n_iter=self.n_iter, algorithm="nearly", transpose=True)
        self.assertEqual(obj.H.shape, self.H_start.T.shape)

if __name__ == '__main__':
    unittest.main()