import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix

from sklearn.utils import shuffle

# randomly shuffle the rows and columns
def shuffle_mat(M):
    M = shuffle(M)
    return shuffle(M.transpose()).transpose()

def subsetAboveDegree(M, col_thres, row_thres):
    M = M.tocsc()
    (m, n) = M.shape
    colSums = np.squeeze(np.array(M.sum(axis=0)))
    rowSums = np.squeeze(np.array(M.sum(axis=1)))
    colValid = colSums > col_thres
    rowValid = rowSums > row_thres
    M1 = M[:, colValid].tocsr()
    M2 = M1[rowValid, :]
    rowFilter = [i for i in range(m) if rowValid[i]]
    colFilter = [i for i in range(n) if colValid[i]]
    return M2, rowFilter, colFilter

def del_block(M:csc_matrix, rowSet ,colSet):
    M = M.tolil()
    (rs, cs) = M.nonzero()
    for i in range(len(rs)):
        if rs[i] in rowSet or cs[i] in colSet:
            M[rs[i], cs[i]] = 0
    return M.tolil()