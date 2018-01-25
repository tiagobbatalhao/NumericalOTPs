import numpy as py
# import qutip as qp
import itertools
import functools

class Pauli():
    """
    Represent a Pauli matrix.
    """
    def __init__(self, label, prefactor = 1):
        label = label.strip()
        assert isinstance(label,str), u"Variable label must be a string"
        assert all([x.lower() in 'izxy' for x in label]), u"Characters must be in the set 'izxy'"
        self.label = label.lower()
        self.n_qubits = len(label)
        prefactor = complex(prefactor)
        self.prefactor = approximate(prefactor.real)+ 1j * approximate(prefactor.imag)

    # def qutip_representation(self, qubit_order='qinfo'):
    #     """
    #     Get a qutip representation of the Pauli operator.
    #     """
    #     assert qubit_order.lower() in ['rigetti','qinfo'], u"Variable 'qubit_order' must be 'rigetti' or 'qinfo'"
    #     if '_qutip_repr' in vars(self) and self._qutip_repr[1] == qubit_order:
    #         return self._qutip_repr[0]
    #     else:
    #         terms = [pauli_matrices[x.lower()] for x in self.label]
    #         if qubit_order == 'rigetti':
    #             rho = qp.tensor(terms[::-1])
    #         elif qubit_order == 'qinfo':
    #             rho = qp.tensor(terms[::+1])
    #         rho = self.prefactor * rho
    #         self._qutip_repr = (rho, qubit_order)
    #         return rho

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            label = self.label.lower()
            prefactor = self.prefactor * other
            return Pauli(label, prefactor)
        elif isinstance(other, Pauli):
            label = []
            prefactor = self.prefactor * other.prefactor
            assert(self.n_qubits == other.n_qubits), u"Incompatible dimensions"
            for labA, labB in zip(self.label,other.label):
                if labA == labB:
                    label.append('i')
                elif labA == 'i':
                    label.append(labB)
                elif labB == 'i':
                    label.append(labA)
                elif labA == 'z' and labB == 'x':
                        label.append('y')
                        prefactor *= +1j
                elif labA == 'z' and labB == 'y':
                        label.append('x')
                        prefactor *= -1j
                elif labA == 'x' and labB == 'z':
                        label.append('y')
                        prefactor *= -1j
                elif labA == 'x' and labB == 'y':
                        label.append('z')
                        prefactor *= +1j
                elif labA == 'y' and labB == 'z':
                        label.append('x')
                        prefactor *= +1j
                elif labA == 'y' and labB == 'x':
                        label.append('z')
                        prefactor *= -1j
            label = ''.join(label)
            new = Pauli(label,prefactor)
            return new
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return self * other
        else:
            raise NotImplementedError

    def trace(self):
        if any([x != 'i' for x in self.label]):
            return 0
        else:
            return 2**self.n_qubits

    def expect(self, state):
        result = self.prefactor
        for sigma, st in zip(self.label, state):
            if sigma != 'i':
                result *= state_definitions.get(sigma,{}).get(st,0)
                if abs(result) < 1e-14:
                    break
        return result

def approximate(number, values=[-1,0,1]):
    for val in values:
        if abs(number-val) < 1e-14:
            return val
    return number

# pauli_matrices = {}
# pauli_matrices['i'] = qp.qeye(2)
# pauli_matrices['x'] = qp.sigmax()
# pauli_matrices['y'] = qp.sigmay()
# pauli_matrices['z'] = qp.sigmaz()
state_definitions = {'z': {'H': +1,'V': -1}, 'X': {'D': +1,'A': -1}, 'Y': {'R': +1,'L': -1}}

class Operator():
    """
    Operator is written as a sum of Pauli matrices and factors.
    """
    def __init__(self, description, n_qubits):
        """
        Example: rho = |0><0| is given by
            Operator(['iii',1],['zzz',1]) / 2
        """
        assert all([len(x[0]) == n_qubits for x in description]), "Incompatible dimensions"
        self.pauli_matrices = [Pauli(x[0],x[1]) for x in description]
        self.n_qubits = n_qubits

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            identity = Operator(('i'*self.n_qubits,1), self.n_qubits)
            return self + identity
        elif isinstance(other, Operator):
            description = {}
            for pauli in self.pauli_matrices + other.pauli_matrices:
                description[pauli.label] = description.get(pauli.label,0) + pauli.prefactor
            nonzero = {x: y for x,y in description.items() if abs(y)>1e-14}
            return Operator(list(nonzero.items()), self.n_qubits)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            description = [(x.label, other*x.prefactor) for x in self.pauli_matrices]
            return Operator(description, self.n_qubits)
        elif isinstance(other, Operator):
            paulis = [x*y for x,y in itertools.product(self.pauli_matrices,other.pauli_matrices)]
            description = {}
            for pauli in paulis:
                description[pauli.label] = description.get(pauli.label,0) + pauli.prefactor
            nonzero = {x: y for x,y in description.items() if abs(y)>1e-14}
            return Operator(list(nonzero.items()), self.n_qubits)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return self * other
        else:
            raise TypeError('Operator can only be right-multiplied by numeric type')

    def __div__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            description = [(x.label, x.prefactor / other) for x in self.pauli_matrices]
            return Operator(description, self.n_qubits)
        else:
            raise TypeError('Operator can only be divided by numeric type')

    # def qutip_representation(self, qubit_order='qinfo'):
    #     return sum([pauli.qutip_representation(qubit_order) for pauli in self.pauli_matrices])


def _tensor_pauli(sigmaA, sigmaB):
    """
    Tensor product of two Pauli matrices
    """
    label = sigmaA.label + sigmaB.label
    prefactor = sigmaA.prefactor * sigmaB.prefactor
    new = Pauli(label, prefactor)
    return new

# def tensor_product(operatorA, operatorB):
#     """
#     Tensor product of two operators
#     """
#     paulis = [_tensor_pauli(x,y) for x,y in itertools.product(operatorA.pauli_matrices,operatorB.pauli_matrices)]
#     description = {}
#     for pauli in paulis:
#         description[pauli.label] = description.get(pauli.label,0) + pauli.prefactor
#     nonzero = {x: y for x,y in description.items() if abs(y)>1e-12}
#     n_qubits = operatorA.n_qubits + operatorB.n_qubits
#     return Operator(list(nonzero.items()), n_qubits)

def tensor_product(*args):
    """
    Tensor product of two operators
    """
    if len(args) == 2:
        paulis = [_tensor_pauli(x,y) for x,y in itertools.product(args[0].pauli_matrices,args[1].pauli_matrices)]
        description = {}
        for pauli in paulis:
            description[pauli.label] = description.get(pauli.label,0) + pauli.prefactor
        nonzero = {x: y for x,y in description.items() if abs(y)>1e-14}
        n_qubits = args[0].n_qubits + args[1].n_qubits
        return Operator(list(nonzero.items()), n_qubits)
    else:
        return functools.reduce(tensor_product, args)
