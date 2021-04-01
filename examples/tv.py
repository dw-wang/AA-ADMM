import sys
sys.path.append('../')

from aa_admm.base_test import *
from aa_admm.solver import *

class TestExamples(BaseTest):
    """ Unit tests for different examples """
    
    def setUp(self):
        np.random.seed(1)
        self.rho = 10
        self.lambda_ = None
        self.J = None
        self.es = None
        self.beta12 = None
        self.rho_M = None
        self.rho_T2 = None
        self.z_true_approx = None
        self.u_true_approx = None
        self.residuals = None
        self.errors = None
        self.eps_abs = 1e-16
        self.eps_rel = 1e-16
        self.maxit = 700
        
    def compute_rho_M(self, ep=1e-5):
        if self.z_true_approx is None or self.u_true_approx is None:
            raise ValueError('Approximate true solution has not been evaluated!')
        nz = self.n - 1
        zu = np.r_[self.z_true_approx, self.u_true_approx]

        # Use finite difference to get approximate gradient of ADMM iterations
        J = np.zeros((2*nz, 2*nz));  # Can be refined to store only n(n+1)/2 entries since Z is symmetric
        for j in range(2*nz):
            h = np.zeros(2*nz)
            h[j] = ep

            zuph = zu + h
            zph = zuph[:nz]
            uph = zuph[nz:]

            # perform one admm step to compute q(z+h)
            xph = prox_sum_squares_affine_base(self.y, self.rho/2, self.D, zph-uph)
            zph = prox_op_l1(self.D @ xph + uph, self.lambda_/self.rho)
            uph = uph + self.D@xph - zph
            zuph = np.r_[zph, uph]

            J[:,j] = (zuph-zu)/ep
        
        self.J = J
        # Compute the spectrum of Jacobian J
        self.es = scipy.linalg.eigvals(J)
        self.rho_M = max(abs(self.es))        
        
    def test_total_variation(self, n=1000):
        # Problem data.
        self.n = n
        self.y = np.random.randn(self.n)
        alpha = 0.001*np.linalg.norm(self.y, np.inf)
        self.lambda_ = alpha

        # Form second difference matrix.
        D = -1*sparse.lil_matrix(sparse.eye(self.n))
        D.setdiag(1, k = 1)
        D = D[:(self.n-1),:].tocsc()
        self.D = D
        
        B = -1 * sparse.eye(self.n-1, format='csc')
        b = np.zeros(self.n-1)
        
        admm_update = [lambda z, u: prox_sum_squares_affine_base(self.y, self.rho/2, self.D, z-u),
                       lambda x, u: prox_op_l1(self.D @ x + u, self.lambda_/self.rho)]

        # Compute results
        self.z_true_approx, self.u_true_approx, _, _, _ = AA_ADMM_ZU(admm_update, D, B, b, 12, self.rho, self.maxit, self.eps_abs, self.eps_rel)
        _, _, r0, e0, t0 = AA_ADMM_ZU(admm_update, D, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, r1, e1, t1 = AA_ADMM_ZU(admm_update, D, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, r2, e2, t2 = AA_ADMM_ZU(admm_update, D, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, r3, e3, t3 = AA_ADMM_ZU(admm_update, D, B, b, 3, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, r5, e5, t5 = AA_ADMM_ZU(admm_update, D, B, b, 5, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, r10, e10, t10 = AA_ADMM_ZU(admm_update, D, B, b, 10, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        
        # Compute beta in sAA(1)-ADMM
        self.compute_rho_M()
        print(self.rho_M)
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M))  # beta for sAA(1)
        self.beta12, self.rho_T2 = opt_sAA2_coeff(self.es)   # beta1, beta2 for sAA(2)
        print(1-np.sqrt(1-self.rho_M))   # rho(T) of sAA(1)
        print(self.rho_T2)               # rho(T2) of sAA(2)
        
        _, _, r_sAA1, e_sAA1, _ = AA_ADMM_ZU(admm_update, D, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx, use_sAA=True, beta=beta)
        _, _, r_sAA2, e_sAA2, _ = AA_ADMM_ZU(admm_update, D, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx, use_sAA=True, beta=self.beta12)
        
        # store residuals results
        self.residuals = [r0, r1, r2, r3, r5, r10, r_sAA1, r_sAA2]
        self.errors = [e0, e1, e2, e3, e5, e10, e_sAA1, e_sAA2]
        self.timings = [t0, t1, t2, t3, t5, t10]
    
    def plot_residuals(self):
        # Plot residuals
        r_sAA = self.residuals[-1]
        rho_ref1 = r_sAA[0] * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(1, 200, 200))
        rho_ref2 = r_sAA[0] * np.power(self.rho_T2, np.linspace(1, 190, 190))
        self.plot_results(self.residuals+[rho_ref1, rho_ref2], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
                linestyles=['-','-', '-', '-', '-', '-', '-', '-', '--', '--'])
        
    def plot_errors(self):
        # Plot residuals
        e_sAA = self.errors[-1]
        rho_ref1 = e_sAA[0] * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(1, 208, 209))
        rho_ref2 = e_sAA[0] * np.power(self.rho_T2, np.linspace(1, 196, 197))
        self.plot_results(self.errors+[rho_ref1, rho_ref2], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
                linestyles=['-','-', '-', '-', '-', '-', '-', '-', '--', '--'],
                pltError=True, \
                yLabel=r'$\sqrt{||z_k-z^*||_2^2+||u_k-u^*||_2^2}$')
        
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
        print(max(abs(eigsT)))
        X = [e.real for e in eigsT]
        Y = [e.imag for e in eigsT]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi'(x^*))$")
        
        # plot eigs of sAA(2) accelerated matrix T2
        nz = self.n - 1
        beta1, beta2 = self.beta12
        I = np.eye(2*nz)
        O = np.zeros((2*nz, 2*nz))
        T2 = np.block([
            [(1+beta1+beta2)*self.J, -beta1*self.J, -beta2*self.J],
            [I,                  O,        O       ],
            [O,                  I,        O       ]
        ])
        eigsT2 = scipy.linalg.eigvals(T2)
        X = [e.real for e in eigsT2]
        Y = [e.imag for e in eigsT2]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi_2'(x^*))$")
        
        #plt.xlim(left=0)
        plt.legend(prop={'size': 10},loc="upper right")
        plt.show()

if __name__ == '__main__':
    tests = TestExamples()
    tests.setUp()
    tests.test_total_variation()
    tests.plot_residuals()
    tests.plot_errors()
    tests.plot_timings()
    tests.plot_eigs()
