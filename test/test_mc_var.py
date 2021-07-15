
from quantum_mc.arithmetic.piecewise_linear_transform import PiecewiseLinearTransform3
import unittest
import numpy as np
from qiskit.test.base import QiskitTestCase
import quantum_mc.calibration.fitting as ft
import quantum_mc.calibration.time_series as ts
from qiskit.circuit.library.arithmetic import weighted_adder
from scipy.stats import multivariate_normal, norm
from qiskit.test.base import QiskitTestCase
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import UniformDistribution, NormalDistribution, LogNormalDistribution
from qiskit.quantum_info import Statevector
import quantum_mc.arithmetic.multiply_add as multiply_add
from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, LinearAmplitudeFunction, IntegerComparator, WeightedAdder

class TestMcVar(QiskitTestCase):

    def test_distribution_load(self):
        """Simple end-to-end test of the (semi-classical) multiply and add building block."""

        correl = ft.get_correl("AAPL", "MSFT")
        
        num_qubits = [3, 3]        
        sigma = correl**2
        bounds = [(-3, 3), (-3, 3)] 
        mu = [0, 0]

        normal = NormalDistribution(num_qubits, mu=mu, sigma=sigma, bounds=bounds)

        transforms = []
        for ticker in ["MSFT", "AAPL"]:
            ((cdf_x, cdf_y), sigma) = ft.get_cdf_data(ticker)
            (x, y) = ft.get_fit_data(ticker, norm_to_rel = False)
            (pl, coeffs) = ft.fit_piecewise_linear(x, y)
            (i_0, i_1, a0, a1, a2, b0, b1, b2, i_to_j, i_to_x, j_to_y) = ft.convert_to_integer(pl, coeffs)
            transforms.append(PiecewiseLinearTransform3(i_0, i_1, a0, a1, a2, b0, b1, b2))

        num_ancillas = transforms[0].num_ancilla_qubits

        qr_input = QuantumRegister(6, 'input') # 2 times 3 registers
        qr_result = QuantumRegister(6, 'result')
        qr_ancilla = QuantumRegister(num_ancillas, 'ancilla')
        output = ClassicalRegister(6, 'output')
        
        circ = QuantumCircuit(qr_input, qr_result, qr_ancilla, output) 
        circ.append(normal, qr_input)

        for i in range(2):
            offset = i * 3
            circ.append(transforms[i], qr_input[offset:offset + 3] + qr_result[:] + qr_ancilla[:])
        
        circ.draw()