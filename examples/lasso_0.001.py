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
        self.u_true_approx = None
        self.J = None
        self.es = None
        self.beta12 = None
        self.rho_M = None
        self.rho_T2 = None
        self.residuals = None
        self.errors = None
        self.eps_abs = 1e-16
        self.eps_rel = 1e-16
        self.maxit = 400  # set to 250 when density = 0.001
    
    def compute_rho_M(self, ep=1e-3):
        n = self.n
        zu = np.r_[self.z_true_approx, self.u_true_approx]

        # Use finite difference to get approximate gradient of ADMM iterations
        J = np.zeros((2*n, 2*n));  # Can be refined to store only n(n+1)/2 entries since Z is symmetric
        for j in range(2*n):
            h = np.zeros(2*n)
            h[j] = ep

            zuph = zu + h
            zph = zuph[:n]
            uph = zuph[n:]

            # perform one admm step to compute q(z+h)
            xph = prox_sum_squares_affine_base(zph-uph, 1/(2*self.rho), self.dataA, self.datab)
            zph = prox_op_l1(xph+uph, self.lambda_/self.rho)
            uph = uph + xph - zph
            zuph = np.r_[zph, uph]

            J[:,j] = (zuph-zu)/ep
        
        self.J = J
        # Compute the spectrum of Jacobian J
        self.es = scipy.linalg.eigvals(J)
        self.rho_M = max(abs(self.es))
    
    def test_lasso(self):
        # Problem data.
        self.m, self.n = 150, 300
        density = 0.001
        np.random.seed(0)
        self.dataA = sparse.random(self.m, self.n, density=density, data_rvs=np.random.randn)
        self.datab = np.random.randn(self.m)
        
        admm_update = [lambda z, u: prox_sum_squares_affine_base(z-u, 1/(2*self.rho), self.dataA, self.datab),
                       lambda x, u: prox_op_l1(x+u, self.lambda_/self.rho)]

        # Set A, B, b
        A = np.eye(self.n)
        B = -np.eye(self.n)
        b = np.zeros(self.n)
        
        # Compute results
        self.z_true_approx, self.u_true_approx, _, _, _ = AA_ADMM_ZU(admm_update, A, B, b, 12, self.rho, self.maxit, self.eps_abs, self.eps_rel)
        _, _, r0, e0, t0 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, r1, e1, t1 = AA_ADMM_ZU(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, r2, e2, t2 = AA_ADMM_ZU(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)       
        _, _, r3, e3, t3 = AA_ADMM_ZU(admm_update, A, B, b, 3, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)       
        _, _, r5, e5, t5 = AA_ADMM_ZU(admm_update, A, B, b, 5, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)       
        _, _, r10, e10, t10 = AA_ADMM_ZU(admm_update, A, B, b, 10, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx)
        
        # Compute beta in sAA(1)-ADMM
        self.compute_rho_M()
        print(self.rho_M)
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M))  # beta for sAA(1)
        self.beta12, self.rho_T2 = opt_sAA2_coeff(self.es)   # beta1, beta2 for sAA(2)
        print(self.beta12)
        print(1-np.sqrt(1-self.rho_M))   # rho(T) of sAA(1)
        print(self.rho_T2)               # rho(T2) of sAA(2)
        
        _, _, r_sAA1, e_sAA1, _ = AA_ADMM_ZU(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx, use_sAA=True, beta=beta)
        _, _, r_sAA2, e_sAA2, _ = AA_ADMM_ZU(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx, use_sAA=True, beta=self.beta12)
        
        # solve with relaxed ADMM (by trying out values in (0, 2), we found 'relax' = 1.9 is the best)
#         _, _, _, e_rx1, t_rx1 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.1, z_true=self.z_true_approx, u_true=self.u_true_approx)
#         _, _, _, e_rx2, t_rx2 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.2, z_true=self.z_true_approx, u_true=self.u_true_approx)
#         _, _, _, e_rx3, t_rx3 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.3, z_true=self.z_true_approx, u_true=self.u_true_approx)
#         _, _, _, e_rx4, t_rx4 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.4, z_true=self.z_true_approx, u_true=self.u_true_approx)
#         _, _, _, e_rx5, t_rx5 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.5, z_true=self.z_true_approx, u_true=self.u_true_approx)
#         _, _, _, e_rx6, t_rx6 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.6, z_true=self.z_true_approx, u_true=self.u_true_approx)
#         _, _, _, e_rx7, t_rx7 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.7, z_true=self.z_true_approx, u_true=self.u_true_approx)
#         _, _, _, e_rx8, t_rx8 = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.8, z_true=self.z_true_approx, u_true=self.u_true_approx)
        _, _, _, e_rx, t_rx = AA_ADMM_ZU(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.9, z_true=self.z_true_approx, u_true=self.u_true_approx)  # best for all cases when density = 0.001, 0.01, 0.06
        
        # store residuals results
        self.errors = [e0, e1, e2, e3, e5, e10, e_rx, e_sAA1, e_sAA2]
        self.timings = [t0, t1, t2, t3, t5, t10, t_rx]
#         self.relax_errors = [e_rx1, e_rx2, e_rx3, e_rx4, e_rx5, e_rx6, e_rx7, e_rx8, e_rx9]
    
#     def plot_aa_admm_errors(self):
#         # Plot errors comparing ADMM and AA-ADMM
#         self.plot_results(self.errors[:7], \
#                 labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'rADMM(1.2)'], \
#                 linestyles=['-','-', '-', '-', '-', '-', '--'],
#                 pltError=True, \
#                 yLabel=r'$\sqrt{||z_k-z^*||_2^2+||u_k-u^*||_2^2}$')
        
    def plot_aa_admm_errors(self):
        e_sAA = self.errors[-1]
        # choose (200, 201) for density = 0.06, (400, 401) for density = 0.01, (125, 126) for density = 0.001
        # change label to back $\rho^*_{sAA(1)}$ when density is not 0.06 (L means lower bound)
        rho_ref1 = (e_sAA[0]+0.6) * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 123, 124))
        rho_ref2 = e_sAA[0] * np.power(self.rho_T2, np.linspace(0, 90, 91))
        # Plot errors comparing ADMM and AA-ADMM
        self.plot_results(self.errors+[rho_ref1, rho_ref2], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'rADMM(1.9)', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
                linestyles=['-', '-', '-', '-', '-', '-', '-.', '-.', '-.', ':', ':'],\
                colors=['k', 'r', 'g', 'b', 'c', 'gray', 'm', 'r', 'g', 'r', 'g'],\
                linewidths=[3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 2.5, 2.5, 2.5, 1.5, 1.5], \
                pltError=True, \
                yLabel=r'$\sqrt{||z_k-z^*||_2^2+||u_k-u^*||_2^2}$',\
                maxIt=250,\
                filename="iters_lasso_0-001.png")
        
        
#     def plot_saa_admm_errors(self):
#         e_sAA = self.errors[-1]
#         # choose (120, 121), (110, 111) for density = 0.001, (245, 246) for density = 0.01
#         rho_ref1 = (e_sAA[0]+0.6) * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 245, 246))
#         rho_ref2 = e_sAA[0] * np.power(self.rho_T2, np.linspace(0, 245, 246))
#         self.plot_results([self.errors[0]]+self.errors[7:]+[rho_ref1, rho_ref2], \
#                 labels=['ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
#                 linestyles=['-','-', '-', '--', '--'],
#                 pltError=True, \
#                 yLabel=r'$\sqrt{||z_k-z^*||_2^2+||u_k-u^*||_2^2}$')
        
    def plot_timings(self):
        self.plot_results(self.errors[:7], ts=self.timings, \
                          labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', \
                                  'AA(10)-ADMM', 'rADMM(1.9)'],
                          linestyles=['-', '-', '-', '-', '-', '-', '-'],\
                          colors=['k', 'r', 'g', 'b', 'c', 'gray', 'm'],\
                          linewidths=[3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5], \
                          pltError=True, \
                          yLabel=r'$\sqrt{||z_k-z^*||_2^2+||u_k-u^*||_2^2}$',\
                          filename='timing_lasso_0-001.png')
        
    def plot_radmm_errors(self):
        # Plot errors comparing ADMM and relaxed-ADMM
        self.plot_results(self.relax_errors, \
                labels=['rADMM(1.1)', 'rADMM(1.2)', 'rADMM(1.3)', 'rADMM(1.4)', 'rADMM(1.5)', 'rADMM(1.6)', 'rADMM(1.7)', 'rADMM(1.8)', 'rADMM(1.9)'], \
                linestyles=['-','-', '-', '-', '-', '-', '-', '-', '-'],
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
        
        # plot eigs of sAA(2) accelerated matrix T2  (do not plot when density = 0.01 and 0.06)
        beta1, beta2 = self.beta12
        I = np.eye(2*self.n)
        O = np.zeros((2*self.n, 2*self.n))
        T2 = np.block([
            [(1+beta1+beta2)*self.J, -beta1*self.J, -beta2*self.J],
            [I,                  O,        O       ],
            [O,                  I,        O       ]
        ])
        eigsT2 = scipy.linalg.eigvals(T2)
        X = [e.real for e in eigsT2]
        Y = [e.imag for e in eigsT2]
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi_2'(X^*))$")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        #plt.xlim(left=0)
        plt.legend(prop={'size': 14},loc="upper right")
        plt.savefig('eigs_lasso_0-001.png')
        plt.show()

if __name__ == '__main__':
    tests = TestExamples()
    tests.setUp()
    tests.test_lasso()
    tests.plot_aa_admm_errors()
#     tests.plot_radmm_errors()
    tests.plot_timings()
    tests.plot_eigs()
