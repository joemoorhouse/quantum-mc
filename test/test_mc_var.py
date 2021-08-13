
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

def get_sims(normal_distribution):
    import numpy as np
    values = normal_distribution._values
    probs = normal_distribution._probabilities
    # we generate a bunch of realisation of values, based 
    upper_bounds = [0.0]
    stop = 0.0
    for val, prob in zip(values, probs):
        stop += prob
        upper_bounds.append(stop)
    
    r = np.random.uniform(low=0.0, high=1.0, size=10)
    indices = np.searchsorted(upper_bounds, r, side='left', sorter=None) - 1

    g1, g2 = np.meshgrid(range(2**3), range(2**3), indexing="ij",)
    i1 = g1.flatten()[indices]
    i2 = g2.flatten()[indices]
    #x = list(zip(*(grid.flatten() for grid in meshgrid)))
    return i1, i2

class TestMcVar(QiskitTestCase):
    
    def test_distribution_load(self):
        """ Test that calculates a cumulative probability from the P&L distribution."""

        correl = ft.get_correl("AAPL", "MSFT")
        
        bounds_std = 3.0
        num_qubits = [3, 3]        
        sigma = correl**2
        bounds = [(-bounds_std, bounds_std), (-bounds_std, bounds_std)] 
        mu = [0, 0]

        # starting point is a multi-variate normal distribution
        normal = NormalDistribution(num_qubits, mu=mu, sigma=sigma, bounds=bounds)

        pl_set = []
        coeff_set = []
        for ticker in ["MSFT", "AAPL"]:
            ((cdf_x, cdf_y), sigma) = ft.get_cdf_data(ticker)
            (x, y) = ft.get_fit_data(ticker, norm_to_rel = False)
            (pl, coeffs) = ft.fit_piecewise_linear(x, y)
            # scale, to apply an arbitrary delta (we happen to use the same value here, but could be different)
            coeffs = ft.scaled_coeffs(coeffs, 1.2)
            pl_set.append(lambda z : ft.piecewise_linear(z, *coeffs))
            coeff_set.append(coeffs)
    
        # calculate the max and min P&Ls
        p_max = max(pl_set[0](bounds_std), pl_set[1](bounds_std))
        p_min = min(pl_set[0](-bounds_std), pl_set[1](-bounds_std))

        # we discretise the transforms and create the circuits
        transforms = []
        i_to_js = []
        for i,ticker in enumerate(["MSFT", "AAPL"]):
            (i_0, i_1, a0, a1, a2, b0, b1, b2, i_to_j, i_to_x, j_to_y) = ft.integer_piecewise_linear_coeffs(coeff_set[i], x_min = -bounds_std, x_max = bounds_std, y_min = p_min, y_max = p_max)
            transforms.append(PiecewiseLinearTransform3(i_0, i_1, a0, a1, a2, b0, b1, b2))
            i_to_js.append(np.vectorize(i_to_j))

        i1, i2 = get_sims(normal)
        j1 = i_to_js[0](i1)
        j2 = i_to_js[1](i2)
        j_tot = j1 + j2

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
