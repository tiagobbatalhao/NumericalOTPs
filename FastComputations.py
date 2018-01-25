import ImplementationOTPs as implement_module
import scipy.sparse.linalg
import gc
import pylab as py

def green_line_prepare(n_copies):
    template = 'Data/PauliAggregate_gate{}_copies{}.txt'
    filename = template.format('0xxx', n_copies)
    rho_0 = implement_module.load_from_file(filename)
    filename = template.format('1xxx', n_copies)
    rho_1 = implement_module.load_from_file(filename)
    rho_diff = rho_0 + (-1)*rho_1
    del rho_0, rho_1, filename, template
    gc.collect()

    sparse_matrix = implement_module.sparse_representation_timed(rho_diff)
    del rho_diff
    gc.collect()

    return sparse_matrix


def lower_red_line_prepare(n_copies, sparse_diagonalization=True, eigenvectors=False):
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

    sparse_matrix = implement_module.sparse_representation_timed(rho_diff)
    del rho_diff
    gc.collect()

    return sparse_matrix


def diagonalize(sparse_matrix, sparse_diagonalization=True, return_eigenvectors=False):
    if sparse_diagonalization:
        if return_eigenvectors:
            eigvals = find_eigenvalues_sparse(sparse_matrix)
            eigvecs = None
        else:
            eigvals, eigvecs = find_eigenvectors_sparse(sparse_matrix)
    else:
        if return_eigenvectors:
            eigvals = find_eigenvalues(sparse_matrix)
            eigvecs = None
        else:
            eigvals, eigvecs = find_eigenvectors(sparse_matrix)
    return eigs

def find_eigenvalues(sparse_matrix):
    return sorted(py.eigvalsh(sparse_matrix.toarray()))
def find_eigenvectors(sparse_matrix):
    return py.eigh(sparse_matrix.toarray())

def find_eigenvalues_sparse(sparse_matrix):
    size_matrix = sparse_matrix.shape[0]
    eigs_top = scipy.sparse.linalg.eigsh(sparse_matrix,
        k=size_matrix/2, which = 'LR', return_eigenvectors = False)
    eigs_bot = scipy.sparse.linalg.eigsh(sparse_matrix,
        k=size_matrix/2, which = 'SR', return_eigenvectors = False)
    return sorted(py.append(eigs_top, eigs_bot))
def find_eigenvectors_sparse(sparse_matrix):
    size_matrix = sparse_matrix.shape[0]
    eigs_top, vecs_top = scipy.sparse.linalg.eigsh(sparse_matrix,
        k=size_matrix/2, which = 'LR', return_eigenvectors = True)
    eigs_bot, vecs_bot = scipy.sparse.linalg.eigsh(sparse_matrix,
        k=size_matrix/2, which = 'SR', return_eigenvectors = True)
    egvals = py.append(eigs_top, eigs_bot)
    egvecs = py.append(vecs_top, vecs_bot, axis=1)
    return egvals, egvecs

def prepare():
    template_green = 'Data/Matrix_green_copies{}.npz'
    template_red = 'Data/Matrix_lowerred_copies{}.npz'
    for n_copies in range(2,8):
        rho = green_line_prepare(n_copies)
        scipy.sparse.save_npz(template_green.format(n_copies), rho)
        rho = lower_red_line_prepare(n_copies)
        scipy.sparse.save_npz(template_red.format(n_copies), rho)
        print('Done for {} copies'.format(n_copies))

def diagonalize():
    template_load_green = 'Data/Matrix_green_copies{}.npz'
    template_load_red = 'Data/Matrix_lowerred_copies{}.npz'
    template_save_green = 'Data/Eigenvalues_green_copies{}.txt'
    template_save_red = 'Data/Eigenvalues_lowerred_copies{}.txt'
    for n_copies in range(2,8):
        rho = scipy.sparse.load_npz(template_load_green.format(n_copies))
        eigvals = find_eigenvalues(rho)
        string = ','.join(['{:f}'.format(x) for x in eigvals])
        with open(template_save_green.format(n_copies), 'w') as f:
            f.write(string)
        del rho, eigvals
        gc.collect()
        print('Done green line for {} copies'.format(n_copies))

        rho = scipy.sparse.load_npz(template_load_red.format(n_copies))
        eigvals = find_eigenvalues(rho)
        string = ','.join(['{:f}'.format(x) for x in eigvals])
        with open(template_save_red.format(n_copies), 'w') as f:
            f.write(string)
        del rho, eigvals
        gc.collect()
        print('Done red line for {} copies'.format(n_copies))



if __name__=='__main__':
    # prepare()
    diagonalize()
