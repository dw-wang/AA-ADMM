import sys
sys.path.append('../')

from aa_admm.base_test import *
from aa_admm.solver import *

class TestExamples(BaseTest):
    """ Unit tests for different examples """
    
    def setUp(self):
        np.random.seed(1)
        self.rho = 10
        self.lambda_ = 1
        self.mu = 0.001
        self.S = None
        self.m = None
        self.n = None
        self.z_true_approx = None
        self.J = None
        self.es = None
        self.beta12 = None
        self.rho_M = None
        self.rho_T2 = None
        self.residuals = None
        self.errors = None
        self.eps_abs = 1e-20
        self.eps_rel = 1e-20
        self.maxit = 175
        
    def update_x(self, z, u):
        Z = z.reshape((self.n, self.n),order='c')
        U = u.reshape((self.n, self.n),order='c')
        
        [es, V] = np.linalg.eigh(self.rho*(Z - U) - self.S);
        xi = (es + np.sqrt(es**2 + 4*self.rho))/(2*self.rho);
        X = V @ np.diag(xi) @ V.T

        return X.reshape(-1,order='C')

    def update_z(self, x, u):
        X = x.reshape((self.n, self.n),order='c')
        U = u.reshape((self.n, self.n),order='c')
        
        Z = X + U

        Z[np.abs(Z) <= self.mu + self.lambda_/self.rho], Z[np.abs(Z) > self.mu + self.lambda_/self.rho] = \
          (self.mu*self.rho/(self.lambda_+self.mu*self.rho)) * Z[np.abs(Z) <= self.mu + self.lambda_/self.rho], \
            Z[np.abs(Z) > self.mu + self.lambda_/self.rho] - (self.lambda_/self.rho) * np.sign(Z[np.abs(Z) > self.mu + self.lambda_/self.rho])

        return Z.reshape(-1,order='C')

    def update_u(self, z):
        Z = z.reshape((self.n, self.n),order='c')
        
        U = np.zeros_like(Z)
        U[np.abs(Z) <= self.mu], U[np.abs(Z) > self.mu] = \
          Z[np.abs(Z) <= self.mu]/self.mu, np.sign(Z[np.abs(Z) > self.mu])
        U *= (self.lambda_/self.rho)

        return U.reshape(-1,order='C')

    def Fx(self, X, Z):
        obj = -np.log(np.linalg.det(X)) + np.trace(X@self.S)
        objZ = np.abs(Z)
        objZ[objZ <= self.mu], objZ[objZ > self.mu] = \
          objZ[objZ <= self.mu]**2/(2*self.mu), objZ[objZ > self.mu] - self.mu/2

        obj += np.sum(objZ)

        return obj
    
    def compute_rho_M(self, ep=5e-3):
        md, nd = self.m, self.n
        z = self.z_true_approx

        # Use finite difference to get approximate gradient of ADMM iterations
        J = np.zeros((nd**2, nd**2));  # Can be refined to store only n(n+1)/2 entries since Z is symmetric
        for j in range(nd**2):
            h = np.zeros(nd**2)
            h[j] = ep

            zph = z + h

            uph = self.update_u(zph)

            # perform one admm step to compute q(z+h)
            xph = self.update_x(zph, uph)
            zph = self.update_z(xph, uph)

            J[:,j] = (zph-z)/ep

        self.J = J
        # Compute the spectrum of Jacobian J
        self.es = scipy.linalg.eigvals(J)
        self.rho_M = max(abs(self.es))
    
    def test_sspinverse_covariance(self):
        n = 40  # number of features
        m = 10*n  # number of samples
        self.m = m
        self.n = n

        # generate a sparse positive definite inverse covariance matrix
        Sinv = np.eye(n)
        np.random.seed(0)
        idx = np.random.choice(n, int(0.2*n))
        idy = np.random.choice(n, int(0.2*n))
        Sinv[idx,idy] = 1
        Sinv = Sinv + Sinv.T   # make symmetric
        w, _ = np.linalg.eigh(Sinv)
        if min(w) < 0:  # make positive definite
            Sinv = Sinv + 1.1*abs(min(w))*np.eye(n)
        S = np.linalg.inv(Sinv)

        # generate Gaussian samples
        D = np.random.multivariate_normal(np.zeros(n), S, m)
        self.S = np.cov(D, rowvar=False)
        
        admm_update = [lambda z, u: self.update_x(z, u),
                       lambda x, u: self.update_z(x, u),
                       lambda z: self.update_u(z)]

        # Set A, B, b
        A = np.eye(self.n**2)
        B = -np.eye(self.n**2)
        b = np.zeros(self.n**2)
        
        # Compute results
        self.z_true_approx, _, _, _ = AA_ADMM_Z(admm_update, A, B, b, 12, self.rho, self.maxit, self.eps_abs, self.eps_rel)
        _, r0, e0, t0 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx)
        _, r1, e1, t1 = AA_ADMM_Z(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx)
        _, r2, e2, t2 = AA_ADMM_Z(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx)       
        _, r3, e3, t3 = AA_ADMM_Z(admm_update, A, B, b, 3, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx)       
        _, r5, e5, t5 = AA_ADMM_Z(admm_update, A, B, b, 5, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx)       
        _, r10, e10, t10 = AA_ADMM_Z(admm_update, A, B, b, 10, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx)
        
        # Compute beta in sAA(1)-ADMM
        self.compute_rho_M()
        print(self.rho_M)
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M))  # beta for sAA(1)
        self.beta12, self.rho_T2 = opt_sAA2_coeff(self.es)   # beta1, beta2 for sAA(2)
        print(1-np.sqrt(1-self.rho_M))   # rho(T) of sAA(1)
        print(self.rho_T2)               # rho(T2) of sAA(2)
        
        _, r_sAA1, e_sAA1, _ = AA_ADMM_Z(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, use_sAA=True, beta=beta)
        _, r_sAA2, e_sAA2, _ = AA_ADMM_Z(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, use_sAA=True, beta=self.beta12)
        
        # store residuals results
        self.residuals = [r0, r1, r2, r3, r5, r10, r_sAA1, r_sAA2]
        self.errors = [e0, e1, e2, e3, e5, e10, e_sAA1, e_sAA2]
        self.timings = [t0, t1, t2, t3, t5, t10]
    
    def plot_residuals(self):
        # Plot residuals
        r_sAA = self.residuals[-1]
        rho_ref1 = r_sAA[0] * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 62, 63))
        rho_ref2 = r_sAA[0] * np.power(self.rho_T2, np.linspace(0, 50, 51))
        self.plot_results(self.residuals+[rho_ref1, rho_ref2], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
                linestyles=['-','-', '-', '-', '-', '-', '-', '-', '--', '--'])
        
    def plot_errors(self):
        # Plot residuals
        e_sAA = self.errors[-1]
        rho_ref1 = e_sAA[0] * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 60, 61))
        rho_ref2 = e_sAA[0] * np.power(self.rho_T2, np.linspace(0, 50, 51))
        self.plot_results(self.errors+[rho_ref1, rho_ref2], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
                linestyles=['-','-', '-', '-', '-', '-', '-', '-', '--', '--'],
                pltError=True)
        
    def plot_timings(self):
        self.plot_results(self.errors[:6], ts=self.timings, \
                          labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM'],
                          linestyles=['-', '-', '-', '-', '-', '-'],
                          pltError=True)
        
    def plot_eigs(self):
        if self.es is None or self.J is None:
            raise ValueError('The Jacobian and spectrum have not been evaluated!')
        # plot eigs of q'
        X = [e.real for e in self.es]
        Y = [e.imag for e in self.es]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(q'(x^*))$")
        
        # plot eigs of accelerated iteration matrix T
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M))
        T = np.block([
            [(1+beta)*self.J,    -beta*self.J],
            [np.eye(self.J.shape[1]), np.zeros(self.J.shape)]
        ])
        eigsT = scipy.linalg.eigvals(T)
        X = [e.real for e in eigsT]
        Y = [e.imag for e in eigsT]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi'(X^*))$")
        
        # plot eigs of sAA(2) accelerated matrix T2
        beta1, beta2 = self.beta12
        I = np.eye(self.n**2)
        O = np.zeros((self.n**2, self.n**2))
        T2 = np.block([
            [(1+beta1+beta2)*self.J, -beta1*self.J, -beta2*self.J],
            [I,                  O,        O       ],
            [O,                  I,        O       ]
        ])
        eigsT2 = scipy.linalg.eigvals(T2)
        X = [e.real for e in eigsT2]
        Y = [e.imag for e in eigsT2]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi_2'(X^*))$")
        
        #plt.xlim(left=0)
        plt.legend(prop={'size': 10},loc="upper right")
        plt.show()

if __name__ == '__main__':
    tests = TestExamples()
    tests.setUp()
    tests.test_sspinverse_covariance()
    tests.plot_residuals()
    tests.plot_errors()
    tests.plot_timings()
    tests.plot_eigs()
    tests.plot_eigs_M()
