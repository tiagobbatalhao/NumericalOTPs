import ImplementationOTPs as implement_module
import scipy.sparse.linalg
import gc
import pylab as py
import time
import pickle

def prepare(n_copies):
    t0 = time.time()

    template = 'Data/PauliAggregate_gate{}_copies{}.txt'
    filename = template.format('00xx', n_copies)
    rho_00 = implement_module.load_from_file(filename)
    filename = template.format('01xx', n_copies)
    rho_01 = implement_module.load_from_file(filename)
    filename = template.format('10xx', n_copies)
    rho_10 = implement_module.load_from_file(filename)
    filename = template.format('11xx', n_copies)
    rho_11 = implement_module.load_from_file(filename)

    template = 'Data/Matrix_rho_{}_copies{}.npz'
    full_rho_00 = implement_module.full_representation(rho_00)
    # scipy.sparse.save_npz(template.format('00', n_copies), full_rho_00)
    full_rho_01 = implement_module.full_representation(rho_01)
    # scipy.sparse.save_npz(template.format('01', n_copies), full_rho_01)
    full_rho_10 = implement_module.full_representation(rho_10)
    # scipy.sparse.save_npz(template.format('10', n_copies), full_rho_10)
    full_rho_11 = implement_module.full_representation(rho_11)
    # scipy.sparse.save_npz(template.format('11', n_copies), full_rho_11)

    t1 = time.time()
    print('Implemented matrices in {:.3f} seconds'.format(t1-t0))

    defined_A_00 = py.dot(full_rho_00 , full_rho_00)
    defined_A_01 = py.dot(full_rho_01 , full_rho_01)
    defined_A_10 = py.dot(full_rho_10 , full_rho_10)
    defined_A_11 = py.dot(full_rho_11 , full_rho_11)

    defined_B = defined_A_00 + defined_A_01 + defined_A_10 + defined_A_11

    t2 = time.time()
    print('Defined A and B in {:.3f} seconds'.format(t2-t1))


    template = 'Data/Matrix_defined_rho_{}_copies{}.pickle'
    with open(template.format('00', n_copies),'wb') as fl:
        pickle.dump(full_rho_00, fl)
    with open(template.format('01', n_copies),'wb') as fl:
        pickle.dump(full_rho_01, fl)
    with open(template.format('10', n_copies),'wb') as fl:
        pickle.dump(full_rho_10, fl)
    with open(template.format('11', n_copies),'wb') as fl:
        pickle.dump(full_rho_11, fl)
    template = 'Data/Matrix_defined_A_{}_copies{}.pickle'
    with open(template.format('00', n_copies),'wb') as fl:
        pickle.dump(defined_A_00, fl)
    with open(template.format('01', n_copies),'wb') as fl:
        pickle.dump(defined_A_01, fl)
    with open(template.format('10', n_copies),'wb') as fl:
        pickle.dump(defined_A_10, fl)
    with open(template.format('11', n_copies),'wb') as fl:
        pickle.dump(defined_A_11, fl)
    template = 'Data/Matrix_defined_B_copies{}.pickle'
    with open(template.format(n_copies),'wb') as fl:
        pickle.dump(defined_B, fl)

    # eigenvalues, eigenvectors = py.eigh(defined_B)
    # eig_invsqrt = [py.sqrt(1./x) if abs(x)>1e-6 else 0 for x in eigenvalues]
    # diagonal = py.diag(eig_invsqrt)
    # invsqrt = py.dot(eigenvectors, py.dot(diagonal), py.conj(eigenvectors).T)



if __name__ == '__main__':
    for n_copies in range(2,8):
        prepare(n_copies)
