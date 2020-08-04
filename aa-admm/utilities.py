# Utilities
from scipy import sparse
import numpy as np
import scipy
from scipy.sparse import diags
from scipy import sparse
from scipy.sparse import linalg

def prox_sum_squares_affine_base(v, t, F, g, method = "lsqr"):
    """Proximal operator of :math:`f(x) = \\|Fx - g\\|_2^2`, where F is a matrix and g is a vector.
    """
#     if F.shape[0] != g.shape[0]:
#        raise ValueError("Dimension mismatch: nrow(F) != nrow(g)")
#     if F.shape[1] != v.shape[0]:
#        raise ValueError("Dimension mismatch: ncol(F) != nrow(v)")

    # Only works on dense vectors.
    if sparse.issparse(g):
        g = g.toarray()[:, 0]
    if sparse.issparse(v):
        v = v.toarray()[:, 0]

    n = v.shape[0]
    if method == "lsqr":
        F = sparse.csr_matrix(F)
        F_stack = sparse.vstack([F, 1/np.sqrt(2*t)*sparse.eye(n)])
        g_stack = np.concatenate([g, 1/np.sqrt(2*t)*v])
        return linalg.lsqr(F_stack, g_stack, atol=1e-16, btol=1e-16)[0]
    elif method == "lstsq":
        if sparse.issparse(F):
            F = F.todense()
        F_stack = np.vstack([F, 1/np.sqrt(2*t)*np.eye(n)])
        g_stack = np.concatenate([g, 1/np.sqrt(2*t)*v])
        return np.linalg.lstsq(F_stack, g_stack, rcond=None)[0]
    else:
        raise ValueError("method must be 'lsqr' or 'lstsq'")
        
        
def prox_sum_squares_base(v, t):
    """Proximal operator of :math:`f(x) = \\sum_i x_i^2`.
    """
    return v / (1.0 + 2*t)

def prox_op_nuclear(X, lambda_):
    """
    Compute the proximal operator of nuclear norm 
    """
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s-lambda_, 0)
    prox = u @ diags(s) @ vh
    
    return prox

def prox_op_l1(X, lambda_):
    """
    Compute the proximal operator of L1 norm
    """
    X = np.sign(X) * np.maximum(abs(X) - lambda_, 0)
    
    return X

def prox_op_lInf(X, lambda_):
    """
    Compute the proximal operator of L_inf norm
    """
    X = lambda_ * np.sign(X) * np.minimum(abs(X), 1)
    
    return X
