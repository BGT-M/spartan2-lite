import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix

from sklearn.utils import shuffle


def shuffle_mat(M):
    M = shuffle(M)
    return shuffle(M.transpose()).transpose()

def del_block(M:csc_matrix, rowSet ,colSet):
    M = M.tolil()
    (rs, cs) = M.nonzero()
    for i in range(len(rs)):
        if rs[i] in rowSet or cs[i] in colSet:
            M[rs[i], cs[i]] = 0
    return M.tolil()