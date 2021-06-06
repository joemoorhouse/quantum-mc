import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy import interpolate
from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UniformDistribution, NormalDistribution, LogNormalDistribution
from qiskit.quantum_info import Statevector

from qiskit.aqua.algorithms import IterativeAmplitudeEstimation

from .probability_distributions import GaussianCopula

def mc_var_circuit():
    mu = [1, 0.9]
    sigma = [[1, -0.2], [-0.2, 1]]

    num_qubits = [3, 2]
    bounds = [(-1, 1), (-1, 1)] 
    
    cdfs, pdfs = prepare_marginals()

    circuit = GaussianCopula(num_qubits, mu=mu, sigma=sigma, bounds=bounds, cdfs = cdfs, pdfs = pdfs)


    state_preparation = get_cdf_circuit(x_eval)
    ae_var = IterativeAmplitudeEstimation(state_preparation=state_preparation,
                                          epsilon=epsilon, alpha=alpha,
                                          objective_qubits=[len(qr_state)])
    result_var = ae_var.run(quantum_instance=Aer.get_backend(simulator), shots=100)

def prepare_marginals():
    # piecewise function for interpolation:
    x = [-4, -1.551, 2.030, 4]
    y = [-11.643, -0.907, 1.363, 5.798]

    pw = interpolate.interp1d(y, x)
    
    def F(x):
        return norm.cdf(pw(x))

    # analytic expression also exists
    def f(x):
        return (F(x + 0.01) - F(x)) / 0.01
    
    Fv = np.vectorize(F)
    fv = np.vectorize(f)

    cdfs = [Fv, Fv]
    pdfs = [fv, fv]

    return (cdfs, pdfs)
