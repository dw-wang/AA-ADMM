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
        self.z_true = None
        self.J = None
        self.es = None
        self.rho_M = None
        self.rho_T2 = None  # spectral radius of sAA(2) matrix
        self.beta12 = None
        self.residuals = None
        self.errors = None
        self.eps_abs = 1e-16
        self.eps_rel = 1e-16
        self.maxit = 200
    
    def test_ridge_regression(self):
        # Problem data.
        self.m, self.n = 150, 300
        density = 0.001
        np.random.seed(0)
        self.dataA = sparse.random(self.m, self.n, density=density, data_rvs=np.random.randn, format='csc')
        self.datab = np.random.randn(self.m)
        
        admm_update = [lambda z, u: prox_sum_squares_affine_base(z-u, 1/(2*self.rho), self.dataA, self.datab),
                       lambda x, u: prox_sum_squares_base(x+u, self.lambda_/self.rho),
                       lambda z:    2*self.lambda_*z/self.rho]

        # compute true solution
        self.z_true = prox_sum_squares_affine_base(sparse.csr_matrix((self.n,1)), 1/(4*self.lambda_), self.dataA, self.datab)
        
        # Set A, B, b
        A = np.eye(self.n)
        B = -np.eye(self.n)
        b = np.zeros(self.n)
        
        # Compute results with AA
        _, r0, e0, _, t0 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, solIsApprox=False)
        _, r1, e1, self.c1, t1 = AA_ADMM_Z(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, solIsApprox=False)
        _, r2, e2, self.c2, t2 = AA_ADMM_Z(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, solIsApprox=False)       
        _, r3, e3, self.c3, t3 = AA_ADMM_Z(admm_update, A, B, b, 3, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, solIsApprox=False)       
        _, r5, e5, _, t5 = AA_ADMM_Z(admm_update, A, B, b, 5, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, solIsApprox=False)       
        _, r10, e10, c10, t10 = AA_ADMM_Z(admm_update, A, B, b, 10, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, solIsApprox=False)
        
        # use relaxation to accelerate (reference suggested 'relax' ~ [1.5, 1.8], turns out 1.9 is better)
        #_, _, e_rx1, _, t_rx1 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.5, z_true=self.z_true, solIsApprox=False)
        #_, _, e_rx2, _, t_rx2 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.6, z_true=self.z_true, solIsApprox=False)
        #_, _, e_rx3, _, t_rx3 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.7, z_true=self.z_true, solIsApprox=False)
        _, _, e_rx, _, t_rx = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.9, z_true=self.z_true, solIsApprox=False)
        
        # Compute beta in sAA(1)-ADMM
        # Iteration matrix M
        P = np.linalg.inv(self.dataA.T.toarray() @ self.dataA.toarray() + self.rho * np.eye(self.n))
        self.J = (self.rho*(self.rho-2*self.lambda_)/(self.rho+2*self.lambda_))*P \
             + (2*self.lambda_/(self.rho+2*self.lambda_))*np.eye(self.n)
        bias = (self.rho/(self.rho+2*self.lambda_)) * P@self.dataA.T@self.datab
        self.es = scipy.linalg.eigvals(self.J)
        self.rho_M =  max(abs(self.es))
        print(self.rho_M)
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M))
        self.beta12, self.rho_T2 = opt_sAA2_coeff(self.es)# beta1, beta2 for sAA(2)
        # self.beta123, self.rho_T3 = opt_sAA3_coeff(self.es, np.arange(0.85,1.0,0.005), np.arange(-0.3,-0.2,0.002), np.arange(0.015,0.03,0.001))  # beta1, beta2, beta3 for sAA(3)
        self.beta123 = (0.955, -0.25, 0.028)
        self.rho_T3 = 0.4837
        print(1-np.sqrt(1-self.rho_M))   # rho(T) of sAA(1)
        print(self.rho_T2)               # rho(T2) of sAA(2)
        print(self.rho_T3)               # rho(T3) of sAA(3)
        
        # solve the problem with sAA(1)-ADMM
        _, r_sAA1, e_sAA1, _, _ = AA_ADMM_Z(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, use_sAA=True, beta=beta, solIsApprox=False)
        _, r_sAA2, e_sAA2, _, _ = AA_ADMM_Z(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, use_sAA=True, beta=self.beta12, solIsApprox=False)
        _, r_sAA3, e_sAA3, _, _ = AA_ADMM_Z(admm_update, A, B, b, 3, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true, use_sAA=True, beta=self.beta123, solIsApprox=False)
        
        # store residuals results
        #self.residuals = [r0, r1, r2, r3, r5, r_sAA1, r_sAA2, r_sAA3]
        self.errors = [e0, e1, e2, e3, e5, e_rx, e_sAA1, e_sAA2, e_sAA3]
        self.timings = [t0, t1, t2, t3, t5, t_rx]
    
    def plot_aa_admm_errors(self):
        # Plot errors comparing ADMM and AA-ADMM
        self.plot_results(self.errors[:6], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'rADMM(1.9)'], \
                linestyles=['-', '-', '-', '-', '-', '--'])
        
    def plot_saa_admm_errors(self):
        # Plot errors comparing ADMM and sAA-ADMM
        e_sAA = self.errors[-1]
        rho_ref1 = (e_sAA[0]+0.9) * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 62, 63))
        rho_ref2 = e_sAA[0] * np.power(self.rho_T2, np.linspace(0, 50, 51))
        rho_ref3 = (e_sAA[0]-0.5) * np.power(self.rho_T3, np.linspace(0, 46, 47))
        self.plot_results([self.errors[0]]+self.errors[6:9]+[rho_ref1, rho_ref2, rho_ref3], \
                labels=['ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', 'sAA(3)-ADMM',\
                       r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$', r'$\rho^*_{sAA(3)}$'], \
                linestyles=['-', '-', '-', '-', '--', '--', '--'])

        
    def plot_timings(self):
        self.plot_results(self.errors[:6], ts=self.timings, \
                          labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'rADMM(1.9)'],
                          linestyles=['-', '-', '-', '-', '-', '--'],
                          pltError=True)
        
    def plot_eigs(self):
        if self.es is None or self.J is None:
            raise ValueError('The Jacobian and spectrum have not been evaluated!')
        # plot eigs of q'
        X = [e.real for e in self.es]
        Y = [e.imag for e in self.es]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(q'(x*))$")
        
        # plot eigs of sAA(1) accelerated iteration matrix T
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M))
        T = np.block([
            [(1+beta)*self.J,    -beta*self.J],
            [np.eye(self.J.shape[1]), np.zeros((self.J.shape[1],self.J.shape[1]))]
        ])
        eigsT = scipy.linalg.eigvals(T)
        X = [e.real for e in eigsT]
        Y = [e.imag for e in eigsT]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi'(X^*))$")
        
        # plot eigs of sAA(2) accelerated matrix T2
        beta1, beta2 = self.beta12
        I = np.eye(self.n)
        O = np.zeros((self.n, self.n))
        T2 = np.block([
            [(1+beta1+beta2)*self.J, -beta1*self.J, -beta2*self.J],
            [I,                  O,        O       ],
            [O,                  I,        O       ]
        ])
        eigsT2 = scipy.linalg.eigvals(T2)
        X = [e.real for e in eigsT2]
        Y = [e.imag for e in eigsT2]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi_2'(X^*))$")
        
        # plot eigs of sAA(2) accelerated matrix T3
        beta1, beta2, beta3 = self.beta123
        T3 = np.block([
            [(1+beta1+beta2+beta3)*self.J, -beta1*self.J, -beta2*self.J, -beta3*self.J],
            [I,                            O,             O,             O],
            [O,                            I,             O,             O],
            [O,                            O,             I,             O]
        ])
        eigsT3 = scipy.linalg.eigvals(T3)
        X = [e.real for e in eigsT3]
        Y = [e.imag for e in eigsT3]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi_3'(X^*))$")
        
        plt.xlim(left=0)
        plt.legend(prop={'size': 10},loc="upper right")
        plt.show()

if __name__ == '__main__':
    tests = TestExamples()
    tests.setUp()
    tests.test_ridge_regression()
    tests.plot_aa_admm_errors()
    tests.plot_saa_admm_errors()
    tests.plot_timings()
    tests.plot_eigs()
