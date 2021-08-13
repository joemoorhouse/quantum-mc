from quantum_mc.arithmetic.piecewise_linear_transform import PiecewiseLinearTransform3
import unittest

import numpy as np
from qiskit.circuit.library.arithmetic import weighted_adder
from scipy.stats import multivariate_normal, norm

from qiskit.test.base import QiskitTestCase
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import UniformDistribution, NormalDistribution, LogNormalDistribution
from qiskit.quantum_info import Statevector
import quantum_mc.arithmetic.multiply_add as multiply_add
from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, LinearAmplitudeFunction, IntegerComparator, WeightedAdder

class TestArithmetic(QiskitTestCase):

    def test_replicate_bug(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
        from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
        from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, LinearAmplitudeFunction, IntegerComparator, WeightedAdder
        from qiskit.visualization import plot_histogram
        from quantum_mc.arithmetic import piecewise_linear_transform


        trans = PiecewiseLinearTransform3(3, 5, 5, 3, 4, 1, 7, 2)
        num_ancillas = trans.num_ancilla_qubits

        qr_input = QuantumRegister(3, 'input')
        qr_result = QuantumRegister(7, 'result')
        qr_ancilla = QuantumRegister(num_ancillas, 'ancilla')
        output = ClassicalRegister(7, 'output')
        
        circ = QuantumCircuit(qr_input, qr_result, qr_ancilla, output) 
        #circ.append(normal, qr_input)

        # put 3 into input
        circ.x(qr_input[0])
        circ.x(qr_input[1])

        # put value 30 in result
        circ.x(qr_result[1])
        circ.x(qr_result[2])
        circ.x(qr_result[3])
        circ.x(qr_result[4])

        #circ.append(trans, qr_input[:] + qr_result[:] + qr_ancilla[:])
        #multiply_add.classical_add_mult(circ, 3, 7, qr_input, qr_result, qr_ancilla)

        multiply_add.classical_mult(circ, 3, qr_input, qr_result, qr_ancilla)

        circ.measure(qr_result, output)

        counts = execute(circ, Aer.get_backend('qasm_simulator'), shots = 128).result().get_counts()
        np.testing.assert_equal(counts['01011'], 128) 
    
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

    def test_adder_subtract(self):
        """Simple end-to-end test of the (semi-classical) multiply and add building block."""

        qr_input = QuantumRegister(3, 'input')
        qr_result = QuantumRegister(5, 'result')
        qr_ancilla = QuantumRegister(5, 'ancilla')
        output = ClassicalRegister(5, 'output')
        circ = QuantumCircuit(qr_input, qr_result, qr_ancilla, output) 

        circ.x(qr_input[0])
        circ.x(qr_input[1])
        circ.x(qr_input[2]) # i.e. load up 7 into register

        add_mult = multiply_add.classical_add_mult(circ, 2, -3, qr_input, qr_result, qr_ancilla)
        #circ.append(cond_add_mult, qr_input[:] + qr_result[:] + qr_ancilla[:]) for the conditional form
        
        circ.measure(qr_result, output)

        # 7 * 2 - 3 = 11: expect 01011
        counts = execute(circ, Aer.get_backend('qasm_simulator'), shots = 128).result().get_counts()
        np.testing.assert_equal(counts['01011'], 128) 

    def test_piecewise_transform(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
        from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
        from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, LinearAmplitudeFunction, IntegerComparator, WeightedAdder
        from qiskit.visualization import plot_histogram
        from quantum_mc.arithmetic import piecewise_linear_transform

        sigma = 1
        low = -3
        high = 3
        mu = 0

        #normal = NormalDistribution(3, mu=mu, sigma=sigma**2, bounds=(low, high))
        
        # our test piece-wise transforms:
        # trans0 if x <= 2, x => 6*x + 7
        # trans1 if 2 < x <= 5, x => x + 17 
        # trans2 if x > 5, x => 3*x + 7
        trans = PiecewiseLinearTransform3(2, 5, 6, 1, 3, 7, 17, 7)
        num_ancillas = trans.num_ancilla_qubits

        qr_input = QuantumRegister(3, 'input')
        qr_result = QuantumRegister(6, 'result')
        qr_ancilla = QuantumRegister(num_ancillas, 'ancilla')
        output = ClassicalRegister(6, 'output')
        
        circ = QuantumCircuit(qr_input, qr_result, qr_ancilla, output) 
        #circ.append(normal, qr_input)

        circ.append(trans, qr_input + qr_result + qr_ancilla)

        circ.measure(qr_result, output)

        counts = execute(circ, Aer.get_backend('qasm_simulator'), shots = 128).result().get_counts()
        np.testing.assert_equal(counts['01011'], 128) 


    def in_progress_test_piecewise_transform(self):
        """Simple end-to-end test of the (semi-classical) multiply and add building block."""

        import numpy as np
        import matplotlib.pyplot as plt

        from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
        from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
        from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, LinearAmplitudeFunction, IntegerComparator, WeightedAdder
        from qiskit.visualization import plot_histogram
        from quantum_mc.arithmetic import multiply_add 

        qr_input = QuantumRegister(3, 'input')
        qr_result = QuantumRegister(6, 'result')
        qr_comp = QuantumRegister(2, 'comparisons')
        qr_ancilla = QuantumRegister(6, 'ancilla')
        qr_comp_anc = QuantumRegister(3, 'cond_ancilla')
        output = ClassicalRegister(6, 'output')
        circ = QuantumCircuit(qr_input, qr_result, qr_comp, qr_ancilla, qr_comp_anc, output) 

        # our test piece-wise transforms:
        # trans0 if x <= 2, x => 6*x + 7
        # trans1 if 2 < x <= 5, x => x + 17 
        # trans2 if x > 5, x => 3*x + 7

        sigma = 1
        low = -3
        high = 3
        mu = 0

        normal = NormalDistribution(3, mu=mu, sigma=sigma**2, bounds=(low, high))
        circ.append(normal, qr_input)

        comp0 = IntegerComparator(num_state_qubits=3, value=3, name = "comparator0") # true if i >= point
        comp1 = IntegerComparator(num_state_qubits=3, value=6, name = "comparator1") # true if i >= point
        trans0 = multiply_add.cond_classical_add_mult(6, 7, qr_input, qr_result, qr_ancilla) 
        trans1 = multiply_add.cond_classical_add_mult(1, 17, qr_input, qr_result, qr_ancilla) 
        trans2 = multiply_add.cond_classical_add_mult(3, 7, qr_input, qr_result, qr_ancilla)  

        circ.append(comp0, qr_input[:] + [qr_comp[0]] + qr_ancilla[0:comp0.num_ancillas])
        circ.append(comp1, qr_input[:] + [qr_comp[1]] + qr_ancilla[0:comp0.num_ancillas])

        # use three additional ancillas to define the ranges
        circ.cx(qr_comp[0], qr_comp_anc[0])
        circ.x(qr_comp_anc[0])
        circ.cx(qr_comp[1], qr_comp_anc[2])
        circ.x(qr_comp_anc[2])
        circ.ccx(qr_comp[0], qr_comp_anc[2], qr_comp_anc[1])

        circ.append(trans0, [qr_comp_anc[0]] + qr_input[:] + qr_result[:] + qr_ancilla[:])
        circ.append(trans1, [qr_comp_anc[1]] + qr_input[:] + qr_result[:] + qr_ancilla[:])
        circ.append(trans2, [qr_comp[1]] + qr_input[:] + qr_result[:] + qr_ancilla[:])

        # can uncompute qr_comp_anc
        # then uncompute the comparators 

        circ.measure(qr_result, output)



        