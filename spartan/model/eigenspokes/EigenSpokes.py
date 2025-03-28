import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import csc_matrix

from .._model import DMmodel

class EigenSpokes(DMmodel):
    def __init__(self, data:csc_matrix):
        self.data = data.asfptype()

    def run(self, k = 0, is_directed = False):

        U, S, Vt = slin.svds(self.data, k+3)
        V = Vt.T
        U, S, V = np.flipud(U), np.flip(S), np.flipud(V)
        
        x = U[:, k] * np.where(np.abs(U[:, k].min()) > np.abs(U[:, k].max()), -1, 1)
        y = V[:, k] * np.where(np.abs(V[:, k].min()) > np.abs(V[:, k].max()), -1, 1)
        
        m, n = U.shape[0], V.shape[0]
        x_thresh = 1 / np.sqrt(m)
        y_thresh = 1 / np.sqrt(n)
        
        x_outliers = np.where(x > x_thresh)[0].tolist()
        y_outliers = np.where(y > y_thresh)[0].tolist()

        return (x_outliers, y_outliers) if is_directed else x_outliers
        
    
    def anomaly_detection(self):
        return self.run()

    def save(self, outpath):
        pass
