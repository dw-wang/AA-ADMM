import sys
sys.path.append('../')

from aa_admm.base_test import *
from aa_admm.solver import *

class TestExamples(BaseTest):
    """ Unit tests for different examples """
    
    def setUp(self):
        np.random.seed(1)
        self.rho = 10
        self.lambda_ = 2
        self.dataA = None
        self.y = None
        self.m = None
        self.n = None
        self.z_true_approx = None
        self.J = None
        self.es = None
        self.beta12 = None
        self.beta123 = None
        self.rho_M = None
        self.rho_T2 = None
        self.rho_T3 = None
        self.residuals = None
        self.errors = None
        self.eps_abs = 1e-16
        self.eps_rel = 1e-16
        self.maxit = 100
        
    def update_x(self, z, u, x0=None):
        # solve the x update
        #   minimize [ -logistic(x_i) + (rho/2)||x_i - z^k + u^k||^2 ]
        # via Newton's method; for a single subsystem only.
        alpha = 0.1
        BETA  = 0.5
        TOLERANCE = 1e-15
        MAX_ITER = 100
        yA = np.diag(np.ravel(self.y)) @ self.dataA
        I = np.eye(self.n+1)

        if (x0 is not None):
            x = x0
        else:
            x = np.zeros(self.n+1)

        C = np.c_[-self.y, -yA]
        f = lambda w: np.average(np.log(1 + np.exp(C@w))) + (self.rho/2)*np.linalg.norm(w - z + u)**2
        for iter in range(MAX_ITER):
            fx = f(x)
            g = C.T @ ( np.exp(C@x) / (1 + np.exp(C@x)) ) / self.m + self.rho*(x - z + u)
            H = C.T @ np.diag(np.ravel( np.exp(C@x) / (1 + np.exp(C@x))**2 )) @ C / self.m + self.rho*I
            dx = np.linalg.solve(-H,g)   # Newton step
            dfx = g.T @ dx  # Newton decrement
            if abs(dfx) < TOLERANCE:
                break

            # backtracking
            t = 1
            while f(x + t*dx) > fx + alpha*t*dfx:
                t = BETA*t

            x = x + t*dx

        return x
        
    def compute_rho_M(self):
        if self.z_true_approx is None:
            raise ValueError('Approximate true solution has not been evaluated!')
        z = self.z_true_approx.copy()
        # Use finite difference to get approximate gradient of ADMM iterations
        ep = 1e-4;
        J = np.zeros((self.n+1, self.n+1));  # Can be refined to store only n(n+1)/2 entries since Z is symmetric
        for j in range(self.n+1):
            h = np.zeros(self.n+1)
            h[j] = ep

            zph = z + h
            zmh = z - h

            uph = 2*self.lambda_*zph/self.rho
            umh = 2*self.lambda_*zmh/self.rho

            # perform one admm step to compute q(z+h)
            xph = self.update_x(zph, uph, z)
            zph = self.rho * (xph+uph) / (2*self.lambda_+self.rho)

            # perform one admm step to compute q(z-h)
            xmh = self.update_x(zmh, umh, z)
            zmh = self.rho * (xmh+umh) / (2*self.lambda_+self.rho)

            J[:,j] = np.ravel((zph-zmh)/2/ep)

        self.J = J
        # Compute the spectrum of Jacobian J
        self.es = scipy.linalg.eigvals(J)
        self.rho_M = max(abs(self.es))
    
    def test_regularized_logistic_regression(self):
        # Problem data.
        self.dataA = np.loadtxt('./data/madelon/madelon_train.data')[::10,::2]
        self.y = np.loadtxt('./data/madelon/madelon_train.labels')[::10]
        self.m, self.n = self.dataA.shape
        
        admm_update = [lambda z, u: self.update_x(z, u),
                       lambda x, u: self.rho * (x+u) / (2*self.lambda_+self.rho),
                       lambda z: 2*self.lambda_*z/self.rho]

        
        # Set A, B, b
        A = np.eye(self.n+1)
        B = -np.eye(self.n+1)
        b = np.zeros(self.n+1)
        
        # Compute results
        self.z_true_approx, _, _, self.c20, _ = AA_ADMM_Z(admm_update, A, B, b, 20, self.rho, self.maxit, self.eps_abs, self.eps_rel)
        _, r0, e0, _, t0 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, solIsApprox=True)
        _, r1, e1, self.c1, t1 = AA_ADMM_Z(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, solIsApprox=True)
        _, r2, e2, self.c2, t2 = AA_ADMM_Z(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, solIsApprox=True)        
        _, r3, e3, self.c3, t3 = AA_ADMM_Z(admm_update, A, B, b, 3, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, solIsApprox=True)        
        _, r5, e5, self.c5, t5 = AA_ADMM_Z(admm_update, A, B, b, 5, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, solIsApprox=True)        
        _, r10, e10, self.c10, t10 = AA_ADMM_Z(admm_update, A, B, b, 10, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, solIsApprox=True)        
        #_, r10, e10, self.c20, _ = AA_ADMM_Z(admm_update, A, B, b, 20, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, solIsApprox=True)
        
        # Compute beta in sAA(1)-ADMM
        self.compute_rho_M()
        print(self.rho_M)
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M))  # beta for sAA(1)
        self.beta12, self.rho_T2 = opt_sAA2_coeff(self.es)   # beta1, beta2 for sAA(2)
#         self.beta123, self.rho_T3 = opt_sAA3_coeff(self.es, np.arange(0.5,0.81,0.005), np.arange(-0.2,0.01,0.005), np.arange(0,0.03,0.001))  # beta1, beta2, beta3 for sAA(3)
        self.beta123 = (0.61, -0.115, 0.009)
        self.rho_T3 = 0.3642705722804524
        print(1-np.sqrt(1-self.rho_M))   # rho(T) of sAA(1)
        print(self.rho_T2)               # rho(T2) of sAA(2)
        print(self.rho_T3)               # rho(T3) of sAA(3)
        
        _, r_sAA1, e_sAA1, _, _ = AA_ADMM_Z(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, use_sAA=True, beta=beta, solIsApprox=True)
        _, r_sAA2, e_sAA2, _, _ = AA_ADMM_Z(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, use_sAA=True, beta=self.beta12, solIsApprox=True)
        _, r_sAA3, e_sAA3, _, _ = AA_ADMM_Z(admm_update, A, B, b, 3, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, use_sAA=True, beta=self.beta123, solIsApprox=True)
        
        # accelerate ADMM by over-relaxation ('relax' = 1.9 gives the best convergence)
#         _, _, e_rx1, _, t_rx1 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.1, z_true=self.z_true_approx, solIsApprox=True)
#         _, _, e_rx2, _, t_rx2 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.2, z_true=self.z_true_approx, solIsApprox=True)
#         _, _, e_rx3, _, t_rx3 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.3, z_true=self.z_true_approx, solIsApprox=True)
#         _, _, e_rx4, _, t_rx4 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.4, z_true=self.z_true_approx, solIsApprox=True)
#         _, _, e_rx5, _, t_rx5 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.5, z_true=self.z_true_approx, solIsApprox=True)
#         _, _, e_rx6, _, t_rx6 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.6, z_true=self.z_true_approx, solIsApprox=True)
#         _, _, e_rx7, _, t_rx7 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.7, z_true=self.z_true_approx, solIsApprox=True)
#         _, _, e_rx8, _, t_rx8 = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.8, z_true=self.z_true_approx, solIsApprox=True)
        _, _, e_rx, _, t_rx = AA_ADMM_Z(admm_update, A, B, b, 0, self.rho, self.maxit, self.eps_abs, self.eps_rel, relaxation=1.9, z_true=self.z_true_approx, solIsApprox=True)
        
        # store residuals results
        self.errors = [e0, e1, e2, e3, e5, e10, e_rx, e_sAA1, e_sAA2, e_sAA3]
        self.timings = [t0, t1, t2, t3, t5, t10, t_rx]
    
    def plot_aa_admm_errors(self):
        # Plot errors comparing ADMM and AA-ADMM
        e_sAA = self.errors[6]
        rho_ref1 = (e_sAA[0]+0.05) * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 40, 41))
        rho_ref2 = (e_sAA[0]+0.2) * np.power(self.rho_T2, np.linspace(0, 36, 37))
        rho_ref3 = (e_sAA[0]+0.03) * np.power(self.rho_T3, np.linspace(0, 31, 32))
        self.plot_results(self.errors + [rho_ref1, rho_ref2, rho_ref3], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', \
                        'rADMM(1.6)', 'sAA(1)-ADMM', 'sAA(2)-ADMM', 'sAA(3)-ADMM', \
                         r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$', r'$\rho^*_{sAA(3)}$'], \
                linestyles=['-', '-', '-', '-', '-', '-', '-.', '-.', '-.', '-.', ':', ':', ':'],\
                colors=['k', 'r', 'g', 'b', 'c', 'gray', 'm', 'r', 'g', 'b', 'r', 'g', 'b'],\
                linewidths=[3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 2.5, 2.5, 2.5, 2.5, 1.5, 1.5, 1.5], \
                pltError=True,\
                maxIt=82,\
                filename='iters_logreg.png')
        
    def plot_saa_admm_errors(self):
        # Plot errors comparing ADMM and sAA-ADMM
        e_sAA = self.errors[-1]
        rho_ref1 = e_sAA[0] * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 40, 41))
        rho_ref2 = e_sAA[0] * np.power(self.rho_T2, np.linspace(0, 40, 41))
        rho_ref3 = e_sAA[0] * np.power(self.rho_T3, np.linspace(0, 31, 32))
        self.plot_results([self.errors[0]]+self.errors[7:]+[rho_ref1, rho_ref2, rho_ref3], \
                labels=['ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', 'sAA(3)-ADMM',\
                       r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$', r'$\rho^*_{sAA(3)}$'], \
                linestyles=['-', '-', '-', '-', '--', '--', '--'],
                pltError=True,
                maxIt=82)
        
#     def plot_radmm_errors(self):
#         # Plot errors comparing ADMM and relaxed-ADMM
#         self.plot_results(self.relax_errors, \
#                 labels=['rADMM(1.1)', 'rADMM(1.2)', 'rADMM(1.3)', 'rADMM(1.4)', 'rADMM(1.5)', 'rADMM(1.6)', 'rADMM(1.7)', 'rADMM(1.8)', 'rADMM(1.9)'], \
#                 linestyles=['-','-', '-', '-', '-', '-', '-', '-', '-'],
#                 pltError=True)
        
    def plot_timings(self):
        self.plot_results(self.errors[:7], ts=self.timings, \
                          labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'rADMM(1.9)'],
                          linestyles=['-', '-', '-', '-', '-', '-', '-'],\
                          colors=['k', 'r', 'g', 'b', 'c', 'gray', 'm'],\
                          linewidths=[3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5], \
                          pltError=True,\
                          maxIt=82,\
                          filename='timing_logreg.png')
        
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
        I = np.eye(self.n+1)
        O = np.zeros((self.n+1, self.n+1))
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
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        plt.xlim(left=0)
        plt.legend(prop={'size': 14},loc="upper right")
        plt.savefig('eigs_logreg.png')
        plt.show()

if __name__ == '__main__':
    tests = TestExamples()
    tests.setUp()
    tests.test_regularized_logistic_regression()
    tests.plot_aa_admm_errors()
#     tests.plot_saa_admm_errors()
    tests.plot_timings()
    tests.plot_eigs()
