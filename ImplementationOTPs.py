"""
Implementation of two-gate OTPs
"""

import numpy as py
# import qutip as qp
import itertools
import functools
import scipy.sparse
import PauliClass as pauli_module
import time

# Definition of mutually anti-commuting Pauli operators
sigmas = []
# sigmas.append(pauli_module.Operator([('zi',1)], 2))
# sigmas.append(pauli_module.Operator([('xi',1)], 2))
# sigmas.append(pauli_module.Operator([('yz',1)], 2))
# sigmas.append(pauli_module.Operator([('yx',1)], 2))
# sigmas.append(pauli_module.Operator([('xi',1)], 2))
# sigmas.append(pauli_module.Operator([('yi',1)], 2))
# sigmas.append(pauli_module.Operator([('zx',1)], 2))
# sigmas.append(pauli_module.Operator([('zy',1)], 2))
sigmas = ['xi', 'yi', 'zx', 'zy']

def get_single_copy_state(gate):
    """
    Produce a single copy state related to a specific gate
    Input: either a binary string (like '0110') or a number (like 7)
    """
    if isinstance(gate, str):
        binary_string = gate[0:4].zfill(4)
        assert all([x in '01' for x in binary_string]), u"Characters must be 0 or 1"
    elif isinstance(gate, int):
        assert (gate >= 0 and gate < 16), u"Variable gate must be between 0 and 15."
        binary_string = bin(gate)[2:].zfill(4)
    else:
        raise TypeError("Variable gate must be string or integer")
    signals = [+1 if x == '0' else -1 if x == '1' else 0 for x in binary_string]
    paulis = [('ii',0.25)]
    for signal, sigma in zip(signals, sigmas):
        paulis.append((sigma, signal * 0.125))
    operator = pauli_module.Operator(paulis, 2)
    return operator

def get_state(gate, n_copies):
    assert isinstance(n_copies,int), "Variable n_copies must be an integer"
    single_copy = get_single_copy_state(gate)
    tensor = pauli_module.tensor_product(*[8*single_copy]*n_copies)
    return tensor

def write_to_file(rho, filename):
    with open(filename, 'w') as fl:
        for pauli in rho.pauli_matrices:
            line = '{:s} {}\n'.format(pauli.label, (pauli.prefactor.real))
            fl.write(line)

def create_all_rhos(n_copies):
    for gate in range(16):
        binary_string = bin(gate)[2:].zfill(4)
        filename = 'Data/PauliExpansion_gate{}_copies{}.txt'.format(binary_string, n_copies)
        rho = get_state(gate, n_copies)
        write_to_file(rho, filename)

def load_from_file(filename):
    description = []
    with open(filename, 'r') as fl:
        for line in fl.readlines():
            split = line.split(' ')
            description.append((split[0],float(split[1])))
    n_qubits = len(description[0][0])
    return pauli_module.Operator(description, n_qubits)


def aggregate(n_copies):
    rho_orig = {}
    for gate in range(16):
        binary_string = bin(gate)[2:].zfill(4)
        filename = 'Data/PauliExpansion_gate{}_copies{}.txt'.format(binary_string, n_copies)
        rho_orig[binary_string] = load_from_file(filename)
    rho_agg = {}
    for label in itertools.product('01x',repeat=4):
        iterables = ['0' if x=='0' else '1' if x=='1' else '01' if x=='x' else '' for x in label]
        summation = []
        for states in itertools.product(*iterables):
            summation.append(rho_orig[''.join(states)])
        rho = functools.reduce(lambda x,y: x+y, summation)
        rho = rho * (1./len(summation))
        rho_agg[''.join(label)] = rho
    return rho_orig, rho_agg

def aggregate_save(n_copies):
    rho_orig, rho_agg = aggregate(n_copies)
    template = 'Data/PauliAggregate_gate{}_copies{}.txt'
    for label, rho in rho_agg.items():
        filename = template.format(label, n_copies)
        print('Wrote to {}'.format(filename))
        write_to_file(rho, filename)


def sparse_representation_pauli(pauli):
    matrices = []
    for i in range(int(len(pauli.label)/2)):
        label = ''.join(pauli.label[2*i:2*i+2])
        matrices.append(sparse_matrices[label])
    tensor = functools.reduce(lambda x,y: scipy.sparse.kron(x,y),matrices)
    tensor = tensor * pauli.prefactor
    return tensor

def sparse_representation_timed(operator):
    t0 = time.time()
    sparse_paulis = [sparse_representation_pauli(x) for x in operator.pauli_matrices]
    t1 = time.time()
    print('Created {} Pauli matrices in {:.3f} seconds'.format(len(sparse_paulis),t1-t0))
    if len(sparse_paulis):
        summation = sparse_paulis[0]
        for counter,pauli in enumerate(sparse_paulis[1:]):
            summation += pauli
            # print('Done {:4d} of {:4d}'.format(counter,len(sparse_paulis)))
        # summation = functools.reduce(lambda x,y: x+y, sparse_paulis)
        t2 = time.time()
        print('Summed {} Pauli matrices in {:.3f} seconds'.format(len(sparse_paulis),t2-t1))
        return summation
    else:
        return None

def full_representation(operator):
    sparse_paulis = [sparse_representation_pauli(x) for x in operator.pauli_matrices]
    full_matrix = sparse_paulis[0].toarray()
    for pauli in sparse_paulis[1:]:
        full_matrix += pauli.toarray()
    return full_matrix



sparse_matrices = {}
sparse_matrices['ii'] = scipy.sparse.csr_matrix(
    py.array([[1,0j,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
sparse_matrices[sigmas[0]] = scipy.sparse.csr_matrix(
    py.array([[0j,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]))
sparse_matrices[sigmas[1]] = scipy.sparse.csr_matrix(
    py.array([[0j,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]]))
sparse_matrices[sigmas[2]] = scipy.sparse.csr_matrix(
    py.array([[0j,1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,-1,0]]))
sparse_matrices[sigmas[3]] = scipy.sparse.csr_matrix(
    py.array([[0j,-1j,0,0],[1j,0,0,0],[0,0,0,1j],[0,0,-1j,0]]))
# sigmas = ['xi', 'yi', 'zx', 'zy']


sparse_matrices['ii'] = scipy.sparse.csr_matrix(
    py.array([[1,0j,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
sparse_matrices['iz'] = scipy.sparse.csr_matrix(
    py.array([[1,0j,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]]))
sparse_matrices['ix'] = scipy.sparse.csr_matrix(
    py.array([[0j,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]))
sparse_matrices['iy'] = scipy.sparse.csr_matrix(
    py.array([[0j,-1j,0,0],[1j,0,0,0],[0,0,0,-1j],[0,0,1,0]]))
sparse_matrices['zi'] = scipy.sparse.csr_matrix(
    py.array([[1,0j,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]))
sparse_matrices['zz'] = scipy.sparse.csr_matrix(
    py.array([[1,0j,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))
sparse_matrices['zx'] = scipy.sparse.csr_matrix(
    py.array([[0j,1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,-1,0]]))
sparse_matrices['zy'] = scipy.sparse.csr_matrix(
    py.array([[0j,-1j,0,0],[1j,0,0,0],[0,0,0,1j],[0,0,-1j,0]]))

sparse_matrices['xi'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]))
sparse_matrices['iz'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]]))
sparse_matrices['xx'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]))
sparse_matrices['xy'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]]))

sparse_matrices['yi'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]]))
sparse_matrices['yz'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,-1j,0],[0,0,0,1j],[1j,0,0,0],[0,-1j,0,0]]))
sparse_matrices['yx'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,0,-1j],[0,0,-1j,0],[0,1j,0,0],[1j,0,0,0]]))
sparse_matrices['yy'] = scipy.sparse.csr_matrix(
    py.array([[0j,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]))



if __name__ == '__main__':
    # for n in range(1,8):
        # create_all_rhos(n)
    for n in range(1,8):
        aggregate_save(n)
