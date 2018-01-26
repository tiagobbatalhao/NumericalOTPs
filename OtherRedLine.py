import ImplementationOTPs as implement_module
import scipy.sparse.linalg
import gc
import pylab as py
import time

def prepare(n_copies):
    template = 'Data/PauliAggregate_gate{}_copies{}.txt'
    filename = template.format('00xx', n_copies)
    rho_00 = implement_module.load_from_file(filename)
    filename = template.format('01xx', n_copies)
    rho_01 = implement_module.load_from_file(filename)
    filename = template.format('10xx', n_copies)
    rho_10 = implement_module.load_from_file(filename)
    filename = template.format('11xx', n_copies)
    rho_11 = implement_module.load_from_file(filename)

    defined_A_00 = rho_00 * rho_00
    defined_A_01 = rho_01 * rho_01
    defined_A_10 = rho_10 * rho_10
    defined_A_11 = rho_11 * rho_11

    defined_B = defined_A_00 + defined_A_01 + defined_A_10 + defined_A_11

    template = 'Data/Matrix_rho_{}_copies{}.npz'
    sparse_rho_00 = implement_module.sparse_representation_timed(rho_00)
    scipy.sparse.save_npz(template.format('00',n_copies),sparse_rho_00)
    del sparse_rho_00, rho_00; gc.collect()
    sparse_rho_01 = implement_module.sparse_representation_timed(rho_01)
    scipy.sparse.save_npz(template.format('01',n_copies),sparse_rho_01)
    del sparse_rho_01, rho_01; gc.collect()
    sparse_rho_10 = implement_module.sparse_representation_timed(rho_10)
    scipy.sparse.save_npz(template.format('10',n_copies),sparse_rho_10)
    del sparse_rho_10, rho_10; gc.collect()
    sparse_rho_11 = implement_module.sparse_representation_timed(rho_11)
    scipy.sparse.save_npz(template.format('11',n_copies),sparse_rho_11)
    del sparse_rho_11, rho_11; gc.collect()

    template = 'Data/Matrix_definedA_{}_copies{}.npz'
    sparse_defined_A_00 = implement_module.sparse_representation_timed(defined_A_00)
    scipy.sparse.save_npz(template.format('00',n_copies),sparse_defined_A_00)
    del sparse_defined_A_00, defined_A_00; gc.collect()
    sparse_defined_A_01 = implement_module.sparse_representation_timed(defined_A_01)
    scipy.sparse.save_npz(template.format('01',n_copies),sparse_defined_A_01)
    del sparse_defined_A_01, defined_A_01; gc.collect()
    sparse_defined_A_10 = implement_module.sparse_representation_timed(defined_A_10)
    scipy.sparse.save_npz(template.format('10',n_copies),sparse_defined_A_10)
    del sparse_defined_A_10, defined_A_10; gc.collect()
    sparse_defined_A_11 = implement_module.sparse_representation_timed(defined_A_11)
    scipy.sparse.save_npz(template.format('11',n_copies),sparse_defined_A_11)
    del sparse_defined_A_11, defined_A_11; gc.collect()

    template = 'Data/Matrix_definedB_copies{}.npz'
    sparse_defined_B = implement_module.sparse_representation_timed(defined_B)
    scipy.sparse.save_npz(template.format(n_copies),sparse_defined_B)
    del sparse_defined_B, defined_B; gc.collect()

if __name__ == '__main__':
    for n_copies in range(2,8):
        prepare(n_copies)
