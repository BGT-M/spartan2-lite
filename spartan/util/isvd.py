import numpy as np
from scipy.sparse.linalg import LinearOperator

def incremental_svd(A, u, s, vh, n_iter=5, k=1):
    """
    Incremental SVD computation without explicitly constructing the residual matrix.
    
    Args:
        A: Original matrix (m x n) or a function that can compute A @ x and A.T @ y
        u: Left singular vectors (m x k) for the first k components
        s: Singular values (length-k vector) for the first k components
        vh: Right singular vectors (k x n) for the first k components
        n_iter: Number of power iterations for approximation
        k: Number of singular vectors to compute (k+1-th component)
        
    Returns:
        u_new: Updated left singular vectors (m x (k+1)) including the (k+1)-th component
        s_new: Updated singular values (length-(k+1) vector
        vh_new: Updated right singular vectors ((k+1) x n)
    """
    m, n = A.shape if hasattr(A, 'shape') else (len(u), vh.shape[1])
    
    def R_matvec(x):
        return A @ x - u @ (np.diag(s) @ (vh @ x))
    
    def R_rmatvec(y):
        return A.T @ y - vh.T @ (np.diag(s) @ (u.T @ y))
    
    # Create linear operator for the residual matrix
    R = LinearOperator((m, n), matvec=R_matvec, rmatvec=R_rmatvec)
    
    # Compute the next singular triplet using power iteration
    u1, s1, vh1 = power_iteration(R, n_iter=n_iter, k=k)
    
    u_new = np.hstack([u, u1])
    s_new = np.concatenate([s, s1])
    vh_new = np.vstack([vh, vh1])
    
    return u_new, s_new, vh_new

def power_iteration(A:LinearOperator, n_iter=5, k=1):
    """
    Power iteration method for computing top-k singular vectors.
    Works with both LinearOperator and regular matrices.
    
    Args:
        A: Input matrix or linear operator
        n_iter: Number of iterations
        k: Number of singular vectors to compute
        
    Returns:
        u: Left singular vectors (m x k)
        s: Singular values (length-k vector)
        vh: Right singular vectors (k x n)
    """
    m, n = A.shape if hasattr(A, 'shape') else A.size
    
    # Initialize random starting vectors
    v = np.random.randn(n, k)
    v, _ = np.linalg.qr(v)
    
    # Power iterations
    for _ in range(n_iter):
        u = A.matvec(v)
        u, _ = np.linalg.qr(u)
        
        v = A.T @ u
        v, _ = np.linalg.qr(v)
    
    # Compute singular values
    Av = A @ v
    s = np.linalg.norm(Av, axis=0)
    u = Av / s
    
    return u, s, v.T