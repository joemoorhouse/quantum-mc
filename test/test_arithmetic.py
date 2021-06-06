import unittest

import numpy as np
from scipy.stats import multivariate_normal, norm

from qiskit.test.base import QiskitTestCase
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import UniformDistribution, NormalDistribution, LogNormalDistribution
from qiskit.quantum_info import Statevector
import quantum_mc.arithmetic.multiply_add as multiply_add
from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, LinearAmplitudeFunction, IntegerComparator, WeightedAdder

class TestArithmetic(QiskitTestCase):

    def test_adder(self):
        """Simple end-to-end test of the (semi-classical) multiply and add building block."""

        qr_input = QuantumRegister(3, 'input')
        qr_result = QuantumRegister(5, 'result')
        qr_ancilla = QuantumRegister(5, 'ancilla')
        output = ClassicalRegister(5, 'output')
        circ = QuantumCircuit(qr_input, qr_result, qr_ancilla, output) 

        circ.x(qr_input[0])
        circ.x(qr_input[1])
        circ.x(qr_input[2]) # i.e. load up 7 into register

        add_mult = multiply_add.classical_add_mult(circ, 2, 3, qr_input, qr_result, qr_ancilla)
        #circ.append(cond_add_mult, qr_input[:] + qr_result[:] + qr_ancilla[:]) for the conditional form
        
        circ.measure(qr_result, output)

        # 7 * 2 + 3 = 17: expect 10001
        counts = execute(circ, Aer.get_backend('qasm_simulator'), shots = 128).result().get_counts()
        np.testing.assert_equal(counts['10001'], 128) 

    def in_progress_test_piecewise_transform(self):
        """Simple end-to-end test of the (semi-classical) multiply and add building block."""

        qr_input = QuantumRegister(3, 'input')
        qr_result = QuantumRegister(5, 'result')
        qr_ancilla = QuantumRegister(5, 'ancilla')
        output = ClassicalRegister(5, 'output')
        circ = QuantumCircuit(qr_input, qr_result, qr_ancilla, output) 

        circ.x(qr_input[0])
        circ.x(qr_input[1])
        circ.x(qr_input[2]) # i.e. load up 7 into register

        #p1 x <= 2
        # 6*x + 7

        #p2 2 < x <= 5
        # x + 17 

        #p2 x > 5
        # 3*x + 7

        comp0 = IntegerComparator(num_state_qubits=3, value=3, name = "comparator") # if true if i >= point
        comp1 = IntegerComparator(num_state_qubits=3, value=6, name = "comparator") # if true if i >= point

        qr_cond = QuantumRegister(3, 'ancilla')

        circ.append(comp0, qr_result[:] + qr_cond[0] + qr_ancilla[0:comp0.num_ancilla_qubits])
        circ.append(comp1, qr_result[:] + qr_cond[0] + qr_ancilla[0:comp0.num_ancilla_qubits])

        trans2 = multiply_add.cond_classical_add_mult(3, 7, qr_input, qr_result, qr_ancilla)   
        circ.append(trans2, comp1[:] + qr_input[:] + qr_result[:] + qr_ancilla[:])


        