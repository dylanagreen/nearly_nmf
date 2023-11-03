import unittest

import numpy as np
from nearly_nmf import nmf

try:
    import cupy as cp
    gpu_available = cp.is_available()
except ImportError:
    cp = None
    gpu_available = False

class TestGPU(unittest.TestCase):
    """
    Test GPU vs. CPU
    """
    def setUp(self):
        self.rng = np.random.default_rng(100921)

        self.nobs = 100
        self.nvar = 20
        self.nvec = 5
        self.X = self.rng.uniform(1,2, size=(self.nvar, self.nobs))
        self.V = self.rng.uniform(1,2, size=self.X.shape)

        self.H0 = self.rng.uniform(1,2, size=(self.nvec, self.nobs))
        self.W0 = self.rng.uniform(1,2, size=(self.nvar, self.nvec))

    @unittest.skipUnless(gpu_available, "requires a GPU")
    def test_get_array_module(self):
        cpu_data = np.arange(5)
        gpu_data = cp.arange(5)

        self.assertIsInstance(cpu_data, np.ndarray)
        self.assertIsInstance(gpu_data, cp.ndarray)
        self.assertEqual(nmf._get_array_module(cpu_data), np)
        self.assertEqual(nmf._get_array_module(gpu_data), cp)
        self.assertEqual(nmf._get_array_module(cpu_data, use_gpu=True), cp)
        self.assertEqual(nmf._get_array_module(cpu_data, use_gpu=False), np)
        self.assertEqual(nmf._get_array_module(gpu_data, use_gpu=True), cp)
        self.assertEqual(nmf._get_array_module(gpu_data, use_gpu=False), np)

    @unittest.skipUnless(gpu_available, "requires a GPU")
    def test_nmf(self):
        """
        Test consistency of CPU and GPU
        """
        #- CPU
        H, W = nmf.fit_NMF(self.X, self.V,
                           H_start=self.H0, W_start=self.W0,
                           n_iter=10)
        #- GPU
        Hg, Wg = nmf.fit_NMF(cp.array(self.X), cp.array(self.V),
                           H_start=self.H0, W_start=self.W0,
                           n_iter=10)

        #- Get GPU data back to CPU for comparisons
        Hgx = Hg.get()
        Wgx = Wg.get()

        self.assertTrue(np.allclose(H, Hgx))
        self.assertTrue(np.allclose(W, Wgx))

    @unittest.skipUnless(gpu_available, "requires a GPU")
    def test_nmf_gpu_option(self):
        """
        Test forcing GPU or not
        """
        for algorithm in ('shift', 'nearly'):
            #- Auto-derive CPU
            H, W = nmf.fit_NMF(self.X, self.V,
                               H_start=self.H0, W_start=self.W0,
                               n_iter=10, algorithm=algorithm)

            self.assertIsInstance(H, np.ndarray)
            self.assertIsInstance(W, np.ndarray)

            #- Auto-derive GPU
            Hg, Wg = nmf.fit_NMF(cp.array(self.X), cp.array(self.V),
                               n_iter=10, algorithm=algorithm)

            self.assertIsInstance(Hg, cp.ndarray)
            self.assertIsInstance(Wg, cp.ndarray)

            #- force CPU vs. GPU, but output type should match input type

            #- Force GPU even if input is CPU
            H, W = nmf.fit_NMF(self.X, self.V,
                               H_start=self.H0, W_start=self.W0,
                               n_iter=10, algorithm=algorithm, use_gpu=True)

            self.assertIsInstance(H, np.ndarray)
            self.assertIsInstance(W, np.ndarray)

            #- Force CPU/GPU to match what input already is, is ok

            H, W = nmf.fit_NMF(self.X, self.V,
                               H_start=self.H0, W_start=self.W0,
                               n_iter=10, algorithm=algorithm, use_gpu=False)

            self.assertIsInstance(H, np.ndarray)
            self.assertIsInstance(W, np.ndarray)

            H, W = nmf.fit_NMF(cp.array(self.X), cp.array(self.V),
                               H_start=cp.array(self.H0),
                               W_start=cp.array(self.W0),
                               n_iter=10, algorithm=algorithm, use_gpu=True)

            self.assertIsInstance(H, cp.ndarray)
            self.assertIsInstance(W, cp.ndarray)

            #- But forcing CPU when inputs are GPU is not supported
            with self.assertRaises(TypeError):
                H, W = nmf.fit_NMF(cp.array(self.X), cp.array(self.V),
                                   H_start=cp.array(self.H0),
                                   W_start=cp.array(self.W0),
                                   n_iter=10, algorithm=algorithm,
                                   use_gpu=False)

    @unittest.skipUnless(gpu_available, "requires a GPU")
    def test_object(self):
        blat = nmf.NMF(self.X, self.V)
        blat.fit()
        self.assertIsInstance(blat.H, np.ndarray)
        self.assertIsInstance(blat.W, np.ndarray)

        blat = nmf.NMF(cp.asarray(self.X), cp.asarray(self.V))
        blat.fit()
        self.assertIsInstance(blat.H, cp.ndarray)
        self.assertIsInstance(blat.W, cp.ndarray)

        blat = nmf.NMF(self.X, self.V, use_gpu=True)
        blat.fit()
        self.assertIsInstance(blat.H, np.ndarray)
        self.assertIsInstance(blat.W, np.ndarray)

        blat = nmf.NMF(self.X, self.V, H_start=self.H0, W_start=self.W0,
                       use_gpu=True)
        blat.fit()
        self.assertIsInstance(blat.H, np.ndarray)
        self.assertIsInstance(blat.W, np.ndarray)

if __name__ == '__main__':
    unittest.main()
