from typing import List, Optional
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import NormalDistribution, LogNormalDistribution, LinearAmplitudeFunction, IntegerComparator, WeightedAdder
import quantum_mc.arithmetic.multiply_add as multiply_add


class PiecewiseLinearTransform3(QuantumCircuit):
    def __init__(self, x0, x1, a0, a1, a2, b0, b1, b2):
        qr_input = QuantumRegister(3, 'input')
        num_result_qubits = 6 # 7
        qr_result = QuantumRegister(num_result_qubits, 'result')
        
        self.num_ancilla_qubits = 2 + num_result_qubits + 3
        qr_ancilla = QuantumRegister(self.num_ancilla_qubits, 'ancilla')

        qr_comp_anc = qr_ancilla[0:2]
        qr_arith_anc = qr_ancilla[2:2 + num_result_qubits]
        qr_range_anc = qr_ancilla[2 + num_result_qubits:2 + num_result_qubits + 3]
        
        super().__init__(qr_input, qr_result, qr_ancilla, name='pwise_lin_trans')

        comp0 = IntegerComparator(num_state_qubits=3, value=x0, name = "comparator0") # true if i >= point
        comp1 = IntegerComparator(num_state_qubits=3, value=x1, name = "comparator1") # true if i >= point
        trans0 = multiply_add.cond_classical_add_mult(a0, b0, qr_input, qr_result, qr_arith_anc) 
        trans1 = multiply_add.cond_classical_add_mult(a1, b1, qr_input, qr_result, qr_arith_anc) 
        trans2 = multiply_add.cond_classical_add_mult(a2, b2, qr_input, qr_result, qr_arith_anc)  

        self.append(comp0.to_gate(), qr_input[:] + [qr_comp_anc[0]] + qr_arith_anc[0:comp0.num_ancillas])
        self.append(comp1.to_gate(), qr_input[:] + [qr_comp_anc[1]] + qr_arith_anc[0:comp1.num_ancillas])

        # use three additional ancillas to define the ranges
        self.cx(qr_comp_anc[0], qr_range_anc[0])
        self.x(qr_range_anc[0])
        self.cx(qr_comp_anc[1], qr_range_anc[2])
        self.x(qr_range_anc[2])
        self.ccx(qr_comp_anc[0], qr_range_anc[2], qr_range_anc[1])

        self.append(trans0, [qr_range_anc[0]] + qr_input[:] + qr_result[:] + qr_arith_anc[:])
        self.append(trans1, [qr_range_anc[1]] + qr_input[:] + qr_result[:] + qr_arith_anc[:])
        self.append(trans2, [qr_comp_anc[1]] + qr_input[:] + qr_result[:] + qr_arith_anc[:])
        
        # uncompute range ancillas
        self.ccx(qr_comp_anc[0], qr_range_anc[2], qr_range_anc[1])
        self.x(qr_range_anc[2])
        self.cx(qr_comp_anc[1], qr_range_anc[2])
        self.x(qr_range_anc[0])
        self.cx(qr_comp_anc[0], qr_range_anc[0])
        
        # uncompute comparator ancillas 
        self.append(comp1.to_gate().inverse(), qr_input[:] + [qr_comp_anc[1]] + qr_arith_anc[0:comp1.num_ancillas])
        self.append(comp0.to_gate().inverse(), qr_input[:] + [qr_comp_anc[0]] + qr_arith_anc[0:comp0.num_ancillas])
        
        
        

