import numpy as np
import scipy as sp
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix

class MultilayerNetwork:
    def __init__(self):
        self.As = [] # within-layer adjacency
        self.Cs = {} # cross-layer adjacenccy
        self.Xs = [] # node feats 
        self.B = lil_matrix((0, 0)) # layer-layer indicator

    @property
    def num_layers(self):
        return len(self.As)

    def add_layer(self, A:csr_matrix):
        self.As.append(A)
        
        new_size = self.num_layers
        self.B.resize((new_size, new_size))
    
    def add_cross_layer(self, layer1:int, layer2:int, C:csr_matrix, is_directed=True):
        assert layer1 < self.num_layers and layer2 < self.num_layers
        self.Cs[(layer1, layer2)] = C
        self.B[layer1, layer2] = 1
        
        if not is_directed:
            self.Cs[(layer2, layer1)] = C.T
            self.B[(layer2, layer1)] = 1        
        
    