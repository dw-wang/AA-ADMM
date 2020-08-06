from aa_admm.utilities import *
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 9, 6
rcParams['text.usetex'] = True

def opt_sAA2_coeff(spectrum):
    beta = ()
    min_rho_mu = np.Inf
    for beta1 in np.arange(-1,1.05,0.05):
        for beta2 in np.arange(-1,1.05,0.05):
            rho_mu = 0
            for mu in spectrum:
                roots = np.roots([1, -(1+beta1+beta2)*mu, beta1*mu, beta2*mu])
                max_root = max(abs(roots))
                if max_root > rho_mu:
                    rho_mu = max_root
            if rho_mu < min_rho_mu:
                beta = (beta1, beta2)
                min_rho_mu = rho_mu
    return beta, min_rho_mu

def opt_sAA3_coeff(spectrum, beta1_range, beta2_range, beta3_range):
    beta = ()
    min_rho_mu = np.Inf
    for beta1 in beta1_range:
        for beta2 in beta2_range:
            for beta3 in beta3_range:
                rho_mu = 0
                for mu in spectrum:
                    roots = np.roots([1, -(1+beta1+beta2+beta3)*mu, beta1*mu, beta2*mu, beta3*mu])
                    max_root = max(abs(roots))
                    if max_root > rho_mu:
                        rho_mu = max_root
                if rho_mu < min_rho_mu:
                    beta = (beta1, beta2, beta3)
                    min_rho_mu = rho_mu
    return beta, min_rho_mu

def AA(R_history):
    """
    Compute the combination coefficients alpha_i in Anderson acceleration, i.e., solve
        argmin sum_{i=0}^m alpha_i r_{k-i},  s.t. sum_{i=0}^{m} alpha_i = 1
    Solve using the equivalent least square problem by eliminating the constraint
    """
    nc = R_history.shape[1];

    # Construct least square matrix
    if (nc == 1):
        c = np.ones(1)
    else:
        Y = R_history[:,1:] - R_history[:,0:-1]
        b = R_history[:,-1]
        q, r = np.linalg.qr(Y)

        z = np.linalg.solve(r, q.T @ b)
        c = np.r_[z[0], z[1:] - z[0:-1], 1 - z[-1]]
    
    return c


def AA_ADMM_Z(admm_update, A, B, b, winsize, rho, maxit, eps_abs, eps_rel, z_true=None, use_sAA=False, beta=None, xzu_init=None, solIsApprox=True):
    """
    Apply Anderson accelerated ADMM (accelerated on variable z) for solving:
        f1(x) + f2(z)  s.t. Ax + Bz = b
        
    admm_update: list of functions for computing x_{k+1}, z_{k+1} and u_{k+1}
    xzu_init:    initial approximation of x, z, u
    rho:         penalty parameter in augmented Lagrangian
    winsize:     window size for AA. When winsize = 0, it is reduced to ADMM without acceleration
    maxit:       maximum number of iterations
    eps_abs:     absolute tolerance
    eps_rel:     relative tolerance
    """
    if xzu_init is None:
        xzu_init = [np.zeros(A.shape[1]), np.zeros(B.shape[1]), np.zeros(A.shape[0])]
    x_old = xzu_init[0].copy()
    z_old = xzu_init[1].copy()
    u_old = xzu_init[2].copy()

    x = admm_update[0](z_old, u_old)
    z = admm_update[1](x, u_old)
    u = admm_update[2](z)
    
    r = np.sqrt(rho*np.linalg.norm(A@x + B@z - b)**2 + rho*np.linalg.norm(B@(z-z_old))**2)
    r_history = [r]
    e_history = []
    c_history = []
    if z_true is not None:
        e_history = [np.linalg.norm(z-z_true)]
    
    Qz = z.copy()
    Rz = z-z_old
    
    z_old = z.copy()
    
    k = 1
    while True:        
        k += 1
        
        x = admm_update[0](z, u)
        z = admm_update[1](x, u)
        
        r = np.sqrt(rho*np.linalg.norm(A@x + B@z - b)**2 + rho*np.linalg.norm(B@(z-z_old))**2)
        r_history.append(r)
        
        acc = r
        acc_history = r_history
        if z_true is not None:
            e = np.linalg.norm(z-z_true)
            e_history.append(e)
            if not solIsApprox:
                acc = e
                acc_history = e_history

        if k >= maxit or acc < eps_abs + eps_rel*acc_history[0]:
            break
        
        # Update memory
        Rz = np.c_[Rz, z-z_old]
        Qz = np.c_[Qz, z]
        if (Rz.shape[1] > winsize+1):
            Rz = Rz[:, 1:]
            Qz = Qz[:, 1:]

        # Apply Anderson acceleration
        if (use_sAA and (beta is not None) and Rz.shape[1] > winsize):
            if winsize == 1:
                cz = np.array([-beta, 1+beta])
            else:
                cz = np.array([-1*e for e in beta[::-1]] + [1+sum(beta)])
        else:
            cz = AA(Rz)
            
        c_history.append(cz)
            
        z = Qz @ cz
        u = admm_update[2](z)
        
        z_old = z.copy()

        
    return z, np.array(r_history), np.array(e_history), c_history

def AA_ADMM_ZU(admm_update, A, B, b, winsize, rho, maxit, eps_abs, eps_rel, z_true=None, u_true=None, use_sAA=False, beta=None, xzu_init=None):
    """
    Apply Anderson accelerated ADMM (accelerated on stacked variable [z; u]) for solving:
        f1(x) + f2(z)  s.t. Ax + Bz = b
        
    admm_update: list of functions for computing x_{k+1}, z_{k+1} and u_{k+1}
    xzu_init:    initial approximation of x, z, u
    rho:         penalty parameter in augmented Lagrangian
    winsize:     window size for AA. When winsize = 0, it is reduced to ADMM without acceleration
    maxit:       maximum number of iterations
    eps_abs:     absolute tolerance
    eps_rel:     relative tolerance
    """
    if xzu_init is None:
        xzu_init = [np.zeros(A.shape[1]), np.zeros(B.shape[1]), np.zeros(A.shape[0])]
    x_old = xzu_init[0].copy()
    z_old = xzu_init[1].copy()
    u_old = xzu_init[2].copy()

    x = admm_update[0](z_old, u_old)
    z = admm_update[1](x, u_old)
    u = A@x + B@z - b + u_old
    
    r = np.sqrt(rho*np.linalg.norm(A@x + B@z - b)**2 + rho*np.linalg.norm(B@(z-z_old))**2)
    r_history = [r]
    e_history = []
    #c_history = []
    if z_true is not None and u_true is not None:
        e_history = [np.sqrt(np.linalg.norm(z-z_true)**2 + np.linalg.norm(u-u_true)**2)]
    
    Q = np.r_[z, u]
    R = np.r_[z-z_old, u-u_old]
    
    z_old = z.copy()
    u_old = u.copy()
    
    k = 1
    while True:        
        k += 1
        
        x = admm_update[0](z, u)
        z = admm_update[1](x, u)
        u = A@x + B@z - b + u
        
        r = np.sqrt(rho*np.linalg.norm(A@x + B@z - b)**2 + rho*np.linalg.norm(B@(z-z_old))**2)
        r_history.append(r)
        # if true solution is provided, use it to stop iteration
        if z_true is not None and u_true is not None:
            e = np.sqrt(np.linalg.norm(z-z_true)**2 + np.linalg.norm(u-u_true)**2)
            e_history.append(e)
        
        if k >= maxit or r < eps_abs + eps_rel*r_history[0]:
            break
        
        # Update memory
        R = np.c_[R, np.r_[z-z_old, u-u_old]]
        Q = np.c_[Q, np.r_[z, u]]
        if (R.shape[1] > winsize+1):
            R = R[:, 1:]
            Q = Q[:, 1:]

        # Apply Anderson acceleration
        if (use_sAA and (beta is not None) and R.shape[1] > winsize):
            if winsize == 1:
                c = np.array([-beta, 1+beta])
            else:
                c = np.array([-1*e for e in beta[::-1]] + [1+sum(beta)])
        else:
            c = AA(R)
        
        #c_history.append(c)
            
        zu = Q @ c
        z = zu[:B.shape[1]]
        u = zu[B.shape[1]:]
        
        z_old = z.copy()
        u_old = u.copy()
        
    return z, u, np.array(r_history), np.array(e_history)
