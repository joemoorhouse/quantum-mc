
from quantum_mc.arithmetic.piecewise_linear_transform import PiecewiseLinearTransform3
import unittest
import numpy as np
from qiskit.test.base import QiskitTestCase
import quantum_mc.calibration.fitting as ft
import quantum_mc.calibration.time_series as ts
from scipy.stats import multivariate_normal, norm
from qiskit.test.base import QiskitTestCase
from qiskit import execute, Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import NormalDistribution
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, IntegerComparator
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem

class TestMcVar(QiskitTestCase):

    def test_distribution_load(self):
        """Simple end-to-end test of the (semi-classical) multiply and add building block."""

        correl = ft.get_correl("AAPL", "MSFT")
        
        num_qubits = [3, 3]        
        sigma = correl**2
        bounds = [(-3, 3), (-3, 3)] 
        mu = [0, 0]

        # starting point is a multi-variate normal distribution
        normal = NormalDistribution(num_qubits, mu=mu, sigma=sigma, bounds=bounds)

        # we apply piecewise transforms to obtain the as-calibrated distributions
        transforms = []
        for ticker in ["MSFT", "AAPL"]:
            ((cdf_x, cdf_y), sigma) = ft.get_cdf_data(ticker)
            (x, y) = ft.get_fit_data(ticker, norm_to_rel = False)
            (pl, coeffs) = ft.fit_piecewise_linear(x, y)
            (i_0, i_1, a0, a1, a2, b0, b1, b2, i_to_j, i_to_x, j_to_y) = ft.convert_to_integer(pl, coeffs)
            transforms.append(PiecewiseLinearTransform3(i_0, i_1, a0, a1, a2, b0, b1, b2))

        num_ancillas = transforms[0].num_ancilla_qubits

        qr_input = QuantumRegister(6, 'input') # 2 times 3 registers
        qr_objective = QuantumRegister(1, 'objective')
        qr_result = QuantumRegister(6, 'result')
        qr_ancilla = QuantumRegister(num_ancillas, 'ancilla')
        #output = ClassicalRegister(6, 'output')
        
        state_preparation = QuantumCircuit(qr_input, qr_objective, qr_result, qr_ancilla) #, output) 
        state_preparation.append(normal, qr_input)

        for i in range(2):
            offset = i * 3
            state_preparation.append(transforms[i], qr_input[offset:offset + 3] + qr_result[:] + qr_ancilla[:])
        
        # to calculate the cdf, we use an additional comparator
        x_eval = 4
        comparator = IntegerComparator(len(qr_result), x_eval + 1, geq=False)
        state_preparation.append(comparator, qr_result[:] + qr_objective[:] + qr_ancilla[0:comparator.num_ancillas])
        
        # now check
        check = False
        if check:
            job = execute(state_preparation, backend=Aer.get_backend('statevector_simulator'))
            var_prob = 0
            for i, a in enumerate(job.result().get_statevector()):
                b = ('{0:0%sb}' % (len(qr_input) + 1)).format(i)[-(len(qr_input) + 1):]
                prob = np.abs(a)**2
                if prob > 1e-6 and b[0] == '1':
                    var_prob += prob
            print('Operator CDF(%s)' % x_eval + ' = %.4f' % var_prob)

        # now do AE

        problem = EstimationProblem(state_preparation=state_preparation,
                            objective_qubits=[len(qr_input)])

        # target precision and confidence level
        epsilon = 0.01
        alpha = 0.05
        qi = QuantumInstance(Aer.get_backend('aer_simulator'), shots=100)
        ae_cdf = IterativeAmplitudeEstimation(epsilon, alpha=alpha, quantum_instance=qi)
        result_cdf = ae_cdf.estimate(problem)

        
        conf_int = np.array(result_cdf.confidence_interval)
        print('Estimated value:\t%.4f' % result_cdf.estimation)
        print('Confidence interval: \t[%.4f, %.4f]' % tuple(conf_int))

        state_preparation.draw()

    #def test_amplitude_estimation(self):
        

        #problem - EstimationProblem()