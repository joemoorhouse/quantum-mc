from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
import numpy as np

# define linear objective function
num_sum_qubits = 5
breakpoints = [0]
slopes = [1]
offsets = [0]
f_min = 0
f_max = 10
c_approx = 0.25

objective = LinearAmplitudeFunction(
    num_sum_qubits,
    slope=slopes,
    offset=offsets,
    # max value that can be reached by the qubit register (will not always be reached)
    domain=(0, 2**num_sum_qubits-1),
    image=(f_min, f_max),
    rescaling_factor=c_approx,
    breakpoints=breakpoints
)

qr_sum = QuantumRegister(5, "sum")
state_preparation = QuantumCircuit(qr_sum) # to complete

# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

# construct amplitude estimation
ae_cdf = IterativeAmplitudeEstimation(state_preparation=state_preparation,
                                      epsilon=epsilon, alpha=alpha,
                                      objective_qubits=[len(qr_sum)])
result_cdf = ae_cdf.run(quantum_instance=Aer.get_backend('qasm_simulator'), shots=100)

# print results
exact_value = 1 # to calculate
conf_int = np.array(result_cdf['confidence_interval'])
print('Exact value:    \t%.4f' % exact_value)
print('Estimated value:\t%.4f' % result_cdf['estimation'])
print('Confidence interval: \t[%.4f, %.4f]' % tuple(conf_int))
