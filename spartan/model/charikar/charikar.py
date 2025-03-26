import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix

from .._model import DMmodel
from ..greedy import greeedy_bipartite, greedy_monopartite

class Charikar(DMmodel):
    def __init__(self, data:csr_matrix):
        self.data = data
    
    def run(self):
        adj = self.data
        res, score = greedy_monopartite(adj)
        return res, score
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass
