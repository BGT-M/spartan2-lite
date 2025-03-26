import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .._model import DMmodel
from ..greedy import greeedy_bipartite

class Fraudar(DMmodel):
    def __init__(self, data:csr_matrix, log="col", bipartite=True):
        self.data = data
        self.log = log
        self.bipartite = bipartite
    
    def run(self):
        adj = self.data.tolil()
        if self == "col":
            weight_matrix = log_weighted_col(adj)
        else:
            weight_matrix = log_weighted_row(adj)
        row, col, score = greeedy_bipartite(weight_matrix)
        if self.bipartite:
            res = sorted(row.union(col))
        else:
            res = [sorted(row), sorted(col)]
        
        return res, score
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass

def log_weighted_col(graph:csr_matrix, eps=5):
    # M: scipy sparse matrix
    m, n = graph.shape
    col_sums = graph.sum(axis=0)
    col_weights = 1.0 / np.log(np.squeeze(col_sums.A) + eps)
    col_diag = lil_matrix((n, n))
    col_diag.setdiag(col_weights)
    return graph * col_diag

def log_weighted_row(graph:csr_matrix, eps=5):
    m, n = graph.shape
    row_sums = graph.sum(axis=1)
    row_weights = 1.0 / np.log(np.squeeze(row_sums.A) + eps)
    row_diag = lil_matrix((m, m))
    row_diag.setdiag(row_weights)
    return graph * row_diag