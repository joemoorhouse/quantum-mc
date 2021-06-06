import unittest
#from ddt import ddt, data, unpack

import numpy as np
from scipy.stats import multivariate_normal, norm

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UniformDistribution, NormalDistribution, LogNormalDistribution
from qiskit.quantum_info import Statevector
from quantum_mc.probability_distributions.gaussian_copula import GaussianCopula 
from quantum_mc.arithmetic.multiply_add import multiply_add

class TestGaussianCopulaDistribution(QiskitTestCase):

    def test_gaussian_copula_normal(self):
        """Test the Gaussian copula circuit against the normal case."""
        
        mu = [0.1, 0.9]
        sigma = [[1, -0.2], [-0.2, 1]]

        num_qubits = [3, 2]
        bounds = [(-1, 1), (-1, 1)] 
        
        def F(x, mean, std = 1):
            return norm.cdf(x, loc = mean, scale = std)

        def f(x, mean, std = 1):
            return norm.pdf(x, loc = mean, scale = std)

        cdfs = [lambda x:F(x, mu[0]), lambda x:F(x, mu[1])]
        pdfs = [lambda x:f(x, mu[0]), lambda x:f(x, mu[1])]

        normal = NormalDistribution(num_qubits, mu=mu, sigma=sigma, bounds=bounds)
        gc_normal = GaussianCopula(num_qubits, cdfs, sigma=sigma, bounds=bounds, pdfs = pdfs)

        sv_normal = Statevector.from_instruction(normal)
        sv_gc_normal = Statevector.from_instruction(gc_normal)

        np.testing.assert_array_almost_equal(sv_normal.data, sv_gc_normal.data)


