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
        self.yA = None
        self.m = None
        self.n = None
        self.z_true_approx = None
        self.u_true_approx = None
        self.J = None
        self.es = None
        self.rho_M = None
        self.residuals = None
        self.errors = None
        self.eps_abs = 1e-16
        self.eps_rel = 1e-16
        self.maxit = 220
        
    def update_x(self, z, u, x0=None):
        # solve the x update
        #   minimize [ -logistic(x_i) + (rho/2)||x_i - z^k + u^k||^2 ]
        # via Newton's method; for a single subsystem only.
        alpha = 0.1
        BETA  = 0.5
        TOLERANCE = 1e-15
        MAX_ITER = 100
        I = np.eye(self.n+1)

        if (x0 is not None):
            x = x0
        else:
            x = np.zeros(self.n+1)

        C = np.c_[-self.y, -self.yA]
        f = lambda w: np.average(np.log(1 + np.exp(C@w))) \
                      + self.lambda_*np.linalg.norm(w)**2 \
                      + (self.rho/2)*np.linalg.norm(w - z + u)**2
        for iter in range(MAX_ITER):
            fx = f(x)
            g = C.T @ ( np.exp(C@x) / (1 + np.exp(C@x)) ) / self.m + 2*self.lambda_*x + self.rho*(x - z + u)
            H = C.T @ np.diag(np.ravel( np.exp(C@x) / (1 + np.exp(C@x))**2 )) @ C / self.m + 2*self.lambda_*I + self.rho*I
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
        
    def compute_rho_M(self, ep = 1e-4):
        if self.z_true_approx is None or self.u_true_approx is None:
            raise ValueError('Approximate true solution has not been evaluated!')
        nx = self.n + 1
        zu = np.r_[self.z_true_approx, self.u_true_approx]

        # Use finite difference to get approximate gradient of ADMM iterations
        J = np.zeros((2*nx, 2*nx));  # Can be refined to store only n(n+1)/2 entries since Z is symmetric
        for j in range(2*nx):
            h = np.zeros(2*nx)
            h[j] = ep

            zuph = zu + h
            zph = zuph[:nx]
            uph = zuph[nx:]

            # perform one admm step to compute q(z+h)
            xph = self.update_x(zph, uph)
            zph = prox_op_lInf(xph+uph, 1/self.rho)
            uph = uph + xph - zph
            zuph = np.r_[zph, uph]

            J[:,j] = (zuph-zu)/ep
        
        self.J = J
        # Compute the spectrum of Jacobian J
        self.es = scipy.linalg.eigvals(J)
        self.rho_M = max(abs(self.es))
    
    def test_constrained_regularized_logistic_regression(self):
        # Problem data.
        self.dataA = np.loadtxt('./data/madelon/madelon_train.data')[::10,::2]
        self.y = np.loadtxt('./data/madelon/madelon_train.labels')[::10]
        self.yA = np.diag(np.ravel(self.y)) @ self.dataA
        self.m, self.n = self.dataA.shape
        
        admm_update = [lambda z, u: self.update_x(z, u),
                       lambda x, u: prox_op_lInf(x+u, 1/self.rho)]

        
        # Set A, B, b
        A = np.eye(self.n+1)
        B = -np.eye(self.n+1)
        b = np.zeros(self.n+1)
        
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
        beta = (1-np.sqrt(1-self.rho_M))/(1+np.sqrt(1-self.rho_M)) # beta for sAA(1)
        self.beta12, self.rho_T2 = opt_sAA2_coeff(self.es)   # beta1, beta2 for sAA(2)
        print(1-np.sqrt(1-self.rho_M))   # rho(T) of sAA(1)
        print(self.rho_T2)               # rho(T2) of sAA(2)
        
        _, _, r_sAA1, e_sAA1, _ = AA_ADMM_ZU(admm_update, A, B, b, 1, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx, use_sAA=True, beta=beta)
        _, _, r_sAA2, e_sAA2, _  = AA_ADMM_ZU(admm_update, A, B, b, 2, self.rho, self.maxit, self.eps_abs, self.eps_rel, z_true=self.z_true_approx, u_true=self.u_true_approx, use_sAA=True, beta=self.beta12)
        
        # store residuals results
        self.residuals = [r0, r1, r2, r3, r5, r10, r_sAA1, r_sAA2]
        self.errors = [e0, e1, e2, e3, e5, e10, e_sAA1, e_sAA2]
        self.timings = [t0, t1, t2, t3, t5, t10]
    
    def plot_residuals(self):
        # Plot residuals
        r_sAA = self.residuals[-1]
        rho_ref1 = r_sAA[0] * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 85, 86))
        rho_ref2 = r_sAA[0] * np.power(self.rho_T2, np.linspace(0, 70, 71))
        self.plot_results(self.residuals+[rho_ref1, rho_ref2], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
                linestyles=['-','-', '-', '-', '-', '-', '-', '-', '--', '--'])
        
    def plot_errors(self):
        # Plot residuals
        e_sAA = self.errors[-1]
        rho_ref1 = e_sAA[0] * np.power((1-np.sqrt(1-self.rho_M)), np.linspace(0, 80, 81))
        rho_ref2 = e_sAA[0] * np.power(self.rho_T2, np.linspace(0, 65, 66))
        self.plot_results(self.errors+[rho_ref1, rho_ref2], \
                labels=['ADMM', 'AA(1)-ADMM', 'AA(2)-ADMM', 'AA(3)-ADMM', 'AA(5)-ADMM', 'AA(10)-ADMM', 'sAA(1)-ADMM', 'sAA(2)-ADMM', r'$\rho^*_{sAA(1)}$', r'$\rho^*_{sAA(2)}$'], \
                linestyles=['-','-', '-', '-', '-', '-', '-', '-', '--', '--'],
                pltError=True,
                yLabel=r'$\sqrt{||z-z^*||_2^2+||u-u^*||_2^2}$',
                maxIt=150)
        
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
        plt.scatter(X, Y, marker='*', label=r"$\sigma(\Psi'(x^*))$")
        
        # plot eigs of sAA(2) accelerated matrix T2
        beta1, beta2 = self.beta12
        I = np.eye(2*self.n+2)
        O = np.zeros((2*self.n+2, 2*self.n+2))
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
    tests.test_constrained_regularized_logistic_regression()
    tests.plot_residuals()
    tests.plot_errors()
    tests.plot_timings()
    tests.plot_eigs()
