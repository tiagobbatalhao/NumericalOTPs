import ImplementationOTPs as implement_module
import scipy.sparse.linalg
import gc
import pylab as py
import time

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
    full_rho_00 = implement_module.sparse_representation_timed(rho_00).toarray()
    full_rho_01 = implement_module.sparse_representation_timed(rho_01).toarray()
    full_rho_10 = implement_module.sparse_representation_timed(rho_10).toarray()
    full_rho_11 = implement_module.sparse_representation_timed(rho_11).toarray()

    t1 = time.time()
    print('Implemented matrices in {:.3f} seconds'.format(t1-t0))

    defined_A_00 = py.dot(full_rho_00 , full_rho_00)
    defined_A_01 = py.dot(full_rho_01 , full_rho_01)
    defined_A_10 = py.dot(full_rho_10 , full_rho_10)
    defined_A_11 = py.dot(full_rho_11 , full_rho_11)

    defined_B = defined_A_00 + defined_A_01 + defined_A_10 + defined_A_11

    t2 = time.time()
    print('Defined A and B in {:.3f} seconds'.format(t2-t1))





if __name__ == '__main__':
    for n_copies in range(2,8):
        prepare(n_copies)
