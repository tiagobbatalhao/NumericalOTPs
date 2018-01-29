import ImplementationOTPs as implement_module
import scipy.sparse.linalg
import scipy.linalg
import gc
import pylab as py
import time
import itertools
import functools

def prepare(n_copies, sparse_diagonalization=True, eigenvectors=False):
    template = 'Data/PauliAggregate_gate{}_copies{}.txt'
    filename = template.format('00xx', n_copies)
    rho_00 = implement_module.load_from_file(filename)
    filename = template.format('01xx', n_copies)
    rho_01 = implement_module.load_from_file(filename)
    filename = template.format('10xx', n_copies)
    rho_10 = implement_module.load_from_file(filename)
    filename = template.format('11xx', n_copies)
    rho_11 = implement_module.load_from_file(filename)
    rho_diff = rho_00 + (-1)*rho_01 + (-1)*rho_10 + rho_11
    del rho_00, rho_01, rho_10, rho_11, filename, template
    gc.collect()
    return rho_diff

def build_candidate_unitary(parameters, n_copies):
    sigmas = [x+y for x,y in itertools.product('izxy',repeat=2)]
    sigmas.pop(0)
    sigmas = [implement_module.sparse_matrices[x].toarray() for x in sigmas]
    hermitean = sum([x*y for x,y in zip(sigmas,parameters)])
    unitary = scipy.linalg.expm( -1j*hermitean )
    unitary = functools.reduce(py.kron, [unitary]*n_copies)
    return unitary

def diagonal_sum(parameters, n_copies, rho_diff):
    unitary = build_candidate_unitary(parameters, n_copies)
    transform = py.dot(unitary, py.dot(rho_diff, py.conj(unitary.T)))
    diagonal = py.diag(transform)
    sumdiag = sum(abs(diagonal))
    return sumdiag

def optimization(n_copies, rho_diff, initial_try=None):
    if initial_try is None:
        initial_try = py.random(15)
    error_function = lambda x: -diagonal_sum(x,n_copies,rho_diff)
    optimize = scipy.optimize.fmin(error_function,initial_try)
    return optimize

def main_copies(n_copies, initial_try=None):
    t0 = time.time()
    rho_diff = prepare(n_copies)
    rho_diff = implement_module.full_representation(rho_diff)
    t1 = time.time()
    print('{} copies, created rho in {:.3f} seconds'.format(n_copies,t1-t0))
    optimal = optimization(n_copies, rho_diff)
    sumdiag = diagonal_sum(optimal, n_copies, rho_diff)
    t2 = time.time()
    print('{} copies, optimized in {:.3f} seconds'.format(n_copies,t2-t1))
    return optimal, sumdiag

def main():
    template = 'Optimization_local_copies{}.txt'
    initial_try = None
    for n_copies in range(2,6):
        opt, sumdiag = main_copies(n_copies, initial_try)
        write = ','.join([str(x) for x in opt])
        write += '\n' + str(sumdiag)
        with open(template.format(n_copies), 'w') as f:
            f.write(write)
        initial_try = opt + 0.1*py.random(15)

if __name__=='__main__':
    main()
