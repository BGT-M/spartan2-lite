import numpy as np
from scipy.sparse import find
from numpy.linalg import norm
from typing import List, Set, Tuple, Union
from .._model import DMmodel
from ..mln import MultilayerNetwork
from scipy.sparse import lil_matrix, csr_matrix

class Destine(DMmodel):
    """Implementation of DESTINE algorithm for cross-layer anomaly detection.
    
    Attributes
    ----------
    learning_rate : float
        Base learning rate for gradient descent
    beta : float
        Backtracking decay factor for Armijo line search
    sigma : float
        Armijo condition coefficient
    max_iter : int
        Maximum optimization iterations
    penalty_weight : float
        Weight for missing links penalty (p in paper)
    selection_vector : np.ndarray
        Current state of the selection vector S
    """
    
    def __init__(self, 
                 data:MultilayerNetwork,
                 gamma:float = 0.5,
                 learning_rate: float = 0.1,
                 penalty_weight: float = 1.0,
                 beta: float = 0.9,
                 sigma: float = 0.01,
                 max_iter: int = 100):
        """
        Parameters
        ----------
        tensor : Tuple
            (A, CR, As) where:
            - A: aggregated diagonal block adjacency matrix
            - CR: Hadamard product of C and R matrices
            - As: list of adjacency matrices per layer
        """
        super().__init__(data, model_name="DESTINE")
        self.As, self.Cs, self.GG = data.As, data.Cs, data.B
        
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.penalty_weight = penalty_weight
        self.beta = beta
        self.sigma = sigma
        self.max_iter = max_iter
        self._initialize_selection_vector()
        
    def _initialize_selection_vector(self) -> None:
        """Initialize selection vector S via degree normalization."""
        self.Asizes = [adj.shape[0] for adj in self.As]
        self.sizeA = sum(self.Asizes)
        self.cumsum_Asizes = np.cumsum([0]+self.Asizes)
        
        self.gammas = {}
        for i in range(len(self.As)):
            ni = self.As[i].shape[0]
            di = self.As[i].sum() / (ni*(ni-1))
            for j in range(len(self.As)):
                nj = self.As[j].shape[0]
                dj = self.As[j].sum() / (nj*(nj-1))
                self.gammas[(i,j)] = (di/dj) ** 2 * self.gamma
        
        self.A = lil_matrix((self.sizeA, self.sizeA))
        for i in range(len(self.As)):
            self.A[self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1], self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1]] = self.As[i]

        self.C = lil_matrix((self.sizeA, self.sizeA))
        for i in range(len(self.As)):
            for j in range(len(self.As)):
                if self.GG == 0:
                    continue
                self.C[self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1], self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1]] = self.Cs[(i,j)]
        
        self.CR = self.C.copy()
        self.CRR = self.C.copy()
        self.C = lil_matrix((self.sizeA, self.sizeA))
        for i in range(len(self.As)):
            for j in range(len(self.As)):
                if self.GG == 0:
                    continue
                
                self.CR[self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1], self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1]] *= self.gammas[(i,j)] ** 0.5
                self.CRR[self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1], self.cumsum_Asizes[i]:self.cumsum_Asizes[i+1]] *= self.gammas[(i,j)]
        
        ss = [G.sum(axis=1)/G.sum(axis=1).max() for G in self.As]
        self.selection_vector = np.concatenate(ss, axis=0)
        self.ones = np.ones(self.selection_vector.shape)
        
    @staticmethod
    def _project_back(v: np.ndarray, 
                     lower: float = 0, 
                     upper: float = 1) -> np.ndarray:
        """Project values into [lower, upper] range."""
        return np.clip(v, lower, upper)
    
    def _compute_gradient(self, 
                          iteration: int, 
                          use_cr: int = -1) -> np.ndarray:
        """Compute combined gradient from Eq.9 and regularization."""
        ss = np.split(self.selection_vector, np.cumsum(self.Asizes)[:-1])
        hat_s = np.concatenate([norm(s, 1)*np.ones(s.shape) for s in ss])
        
        # Within-layer gradient
        grad_ld = (2 * self.penalty_weight * (hat_s - self.selection_vector) 
                   - 2 * (1 + self.penalty_weight) * self.A.dot(self.selection_vector))
        
        # Cross-layer regularization gradient
        grad_lr = 0
        if iteration > use_cr:
            s = self.selection_vector
            grad_lr = (np.multiply(s, self.CR.dot(8*np.multiply(s, s) + 2*self.ones - 8*s)) 
                     + self.CR.dot(2*s - 4*np.multiply(s, s)))
            
        return grad_ld + grad_lr
    
    def _armijo_line_search(self, 
                          gradient: np.ndarray, 
                          current_loss: float,
                          iteration: int,
                          use_cr: int) -> float:
        """Adaptive learning rate via Armijo condition."""
        lr = self.learning_rate
        while True:
            s_new = self._project_back(self.selection_vector - lr * gradient)
            new_loss = self._compute_loss(s_new, iteration, use_cr)
            
            diff = s_new - self.selection_vector
            armijo_condition = (new_loss - current_loss <= 
                               self.sigma * gradient.T.dot(diff)[0, 0])
            
            if armijo_condition:
                return lr / np.sqrt(self.beta)
            lr *= self.beta
    
    def _compute_loss(self, 
                     s: np.ndarray, 
                     iteration: int,
                     use_cr: int) -> float:
        """Compute combined optimization objective."""
        ss = np.split(s, np.cumsum(self.Asizes)[:-1])
        within_loss = (self.penalty_weight * sum(norm(si, 1)**2 - norm(si, 2)**2 for si in ss)
                      - (1 + self.penalty_weight) * s.T.dot(self.A.dot(s)))
        
        if iteration <= use_cr:
            return within_loss
            
        # Cross-layer regularization loss
        row_idx, col_idx, data = find(self.CR)
        ones_s = self.ones - s
        cr_loss = norm(data - np.multiply(data, (np.multiply(s[row_idx], s[col_idx]) + 
                     np.multiply(ones_s[row_idx], ones_s[col_idx])).flatten(), 2)**2)
        
        return within_loss + cr_loss
    
    def run(self, 
           start_armijo: int = -1, 
           use_cr: int = -1,
           **kwargs) -> List[Set[int]]:
        """Execute DESTINE optimization and return detected anomalies.
        
        Returns
        -------
        List[Set[int]]
            Detected node indices per layer with selection score ≥0.5
        """
        for it in range(self.max_iter):
            gradient = self._compute_gradient(it, use_cr)
            current_loss = self._compute_loss(self.selection_vector, it, use_cr)
            
            if it > start_armijo:
                self.learning_rate = self._armijo_line_search(
                    gradient, current_loss, it, use_cr)
                
            self.selection_vector = self._project_back(
                self.selection_vector - self.learning_rate * gradient)
            
        return self._get_results()
    
    def _get_results(self) -> List[Set[int]]:
        """Convert selection vector to detected node sets."""
        results = []
        for si in np.split(self.selection_vector, np.cumsum(self.Asizes)[:-1]):
            results.append({j for j, val in enumerate(si.flatten()) if val >= 0.5})
            
        return results