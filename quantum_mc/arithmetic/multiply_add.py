from math import pi
from qiskit import QuantumRegister, QuantumCircuit, AncillaRegister
from .qft import qft, iqft, cqft, ciqft, ccu1
from .arithmetic import cadd, full_qr, add_ripple_in_place, add_ripple_in_place_cq

def classical_add_mult(circ, a, b, qr_in, qr_res, qr_anc):
    """qr_res = qr_res + a * qr_in + b where a and b are integers"""
    classical_mult(circ, a, qr_in, qr_res, qr_anc)
    classical_add(circ, b, qr_res, qr_anc)

def classical_mult(circ, a, qr_in, qr_res, qr_anc):
    """qr_res = a * qr_in where a is an integer and qr_in is a quantum register
        res must have at least na + nin qubits where na is the number of bits required to represent a and nin the number of qubits in qr_in
     """
    l_a = _to_bool_list(a)
    na = len(l_a)
    nin = len(qr_in)
    for i in range (0, na):
        if l_a[i] == 1:
            add_ripple_in_place(circ, qr_in, _sub_qr(qr_res, i, nin + i), qr_anc, nin)


def cond_classical_add_mult(a, b, qr_in, qr_res, qr_anc):
    """qr_res = qr_res + a* qr_in + b if control is set where a and b are integers"""
    temp_qr_in = QuantumRegister(len(qr_in))
    temp_qr_res = QuantumRegister(len(qr_res))
    temp_qr_anc = AncillaRegister(len(qr_anc))
    temp_circuit = QuantumCircuit(temp_qr_in, temp_qr_res, temp_qr_anc, name = "inplace_mult_add")
    classical_add_mult(temp_circuit, a, b, temp_qr_in, temp_qr_res, temp_qr_anc)
    temp_circuit = temp_circuit.control(1) 
    return temp_circuit


def classical_add(circ, a, qr_b, qr_anc):
    """qr_b = a + qr_b where a is an integer and qr_b is a quantum register """
    nb = len(qr_b)
    l_a = _to_bool_list(a)
    if len(l_a) > nb - 1:
        raise Exception("number of classical integer bits cannot exceed number of register qubits - 1")
    l_a = l_a + [0 for i in range(nb - len(l_a))] # pad with zerp
    add_ripple_in_place_cq(circ, l_a, qr_b, qr_anc, nb - 1)

def _to_bool_list(a):
    s = a
    res = []
    while (s > 0):
        res.append(s & 1)
        s = s >> 1
    return res

# Take a subset of a quantum register from index x to y, inclusive.
def _sub_qr(qr, x, y): 
    sub = []
    for i in range (x, y + 1):
        sub = sub + [(qr[i])]
    return sub

def scalar_mult_plot(control, a, b, c, anc, na, nb):
   qa = QuantumRegister(len(a), name = "input")
   qc = QuantumRegister(len(c), name = "output")
   qanc = AncillaRegister(len(anc), name = "ancilla")
   tempCircuit = QuantumCircuit(qa, qc, qanc)
   #scalar_mult(tempCircuit, qa, b, qc, qanc, na, nb)
   return tempCircuit