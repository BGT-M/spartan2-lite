import numpy as np
from scipy.sparse.linalg import LinearOperator

def incremental_svd(A, u, s, vh, n_iter=5):
    """
    增量式SVD - 不显式计算残差矩阵
    
    参数:
    A: 原始矩阵 (m x n) 或能计算A @ x和A.T @ y的函数
    u: 前k个左奇异向量 (m x k)
    s: 前k个奇异值 (长度为k的向量)
    vh: 前k个右奇异向量 (k x n)
    k: 当前已知的奇异值数量
    n_iter: 幂迭代次数
    
    返回:
    u_new: 包含第k+1个左奇异向量的矩阵 (m x (k+1))
    s_new: 包含第k+1个奇异值的向量 (长度为k+1)
    vh_new: 包含第k+1个右奇异向量的矩阵 ((k+1) x n)
    """
    m, n = A.shape if hasattr(A, 'shape') else (len(u), vh.shape[1])
    
    def R_matvec(x):
        return A @ x - u @ (np.diag(s) @ (vh @ x))
    
    def R_rmatvec(y):
        return A.T @ y - vh.T @ (np.diag(s) @ (u.T @ y))
    
    R = LinearOperator((m, n), matvec=R_matvec, rmatvec=R_rmatvec)
    u1, s1, vh1 = power_iteration(R, n_iter=n_iter)
    
    u_new = np.hstack([u, u1])
    s_new = np.concatenate([s, s1])
    vh_new = np.vstack([vh, vh1])
    
    return u_new, s_new, vh_new

def power_iteration(A, n_iter=5, k=1):
    """
    幂迭代法计算矩阵的前k个奇异向量，适用于LinearOperator或普通矩阵
    """
    m, n = A.shape if hasattr(A, 'shape') else A.size
    v = np.random.randn(n, k)
    v, _ = np.linalg.qr(v)
    
    for _ in range(n_iter):
        # 计算A^T A v ≈ σ^2 v
        u = A.matvec(v)
        u, _ = np.linalg.qr(u)
        
        v = A.T @ u
        v, _ = np.linalg.qr(v)
    
    # 计算奇异值
    Av = A @ v
    s = np.linalg.norm(Av, axis=0)
    u = Av / s
    
    return u, s, v.T