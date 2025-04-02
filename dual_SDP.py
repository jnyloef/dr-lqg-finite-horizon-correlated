import numpy as np
import cvxpy as cp

class SDPProblem:
    def __init__(self, n_x=5, n_y=3, n_u=2, T=10):
        self.n_x = n_x
        self.n_y = n_y
        self.n_u = n_u
        self.T = T
        
        self.N_xi = (T + 1) * n_x
        self.N_u = T * n_u
        self.N_y = T * n_y
        self.N_xi_full = n_x + T * (n_x + n_y)
        
        self.initialize_matrices()
        self.define_variables()
        self.define_constraints()
        self.define_objective()
    
    def initialize_matrices(self):
        self.A = np.random.randn(self.n_x, self.n_x)
        self.B = np.random.randn(self.n_x, self.n_u)
        self.C = np.random.randn(self.n_y, self.n_x)
        
        self.H = np.zeros((self.N_xi, self.N_u))
        for t in range(1, self.T + 1):
            for k in range(t):
                self.H[t * self.n_x:(t + 1) * self.n_x, k * self.n_u:(k + 1) * self.n_u] = np.linalg.matrix_power(self.A, t - 1 - k) @ self.B
        
        self.D = np.zeros((self.N_xi, self.N_xi_full))
        self.D[:self.n_x, :self.n_x] = np.eye(self.n_x)
        for t in range(1, self.T + 1):
            self.D[t * self.n_x:(t + 1) * self.n_x, :self.n_x] = np.linalg.matrix_power(self.A, t)
            for k in range(t):
                self.D[t * self.n_x:(t + 1) * self.n_x, self.n_x + k * (self.n_x + self.n_y):self.n_x + (k + 1) * (self.n_x + self.n_y) - self.n_y] = np.linalg.matrix_power(self.A, t - 1 - k)
        
        self.C_block = np.kron(np.eye(self.T), self.C)
        self.E = np.zeros((self.N_y, self.N_xi_full))
        for t in range(self.T):
            self.E[t * self.n_y:(t + 1) * self.n_y, self.n_x + t * (self.n_x + self.n_y) + self.n_x:self.n_x + (t + 1) * (self.n_x + self.n_y)] = np.eye(self.n_y)
        
        self.F = self.C_block @ self.D + self.E
        
        A_Q = np.random.randn(self.N_xi, self.N_xi)
        self.Q = A_Q.T @ A_Q
        
        B_R = np.random.randn(self.N_u, self.N_u)
        epsilon = 1e-3
        self.R = B_R.T @ B_R + epsilon * np.eye(self.N_u)
        
        C_Sigma = np.random.randn(self.N_xi, self.N_xi)
        self.Sigma_hat = C_Sigma.T @ C_Sigma + epsilon * np.eye(self.N_xi)
        
        self.lambda_min = np.min(np.linalg.eigvals(self.Sigma_hat))
        self.rho = 1.0
        self.mu_hat = np.random.randn(self.N_xi, 1)
    
    def define_variables(self):
        self.mu = cp.Variable((self.N_xi, 1))
        self.M = cp.Variable((self.N_xi, self.N_xi), PSD=True)
        self.N = cp.Variable((self.N_xi, self.N_xi), PSD=True)
        self.L = cp.Variable((self.N_xi, self.N_xi))
        self.Lambda = cp.Variable((self.N_u, self.N_y))
        self.K = cp.Variable((self.N_u, self.N_u), PSD=True)
    
    def define_constraints(self):
        self.constraints = []
        
        self.constraints.append(cp.bmat([
            [self.K, self.H.T @ self.Q @ self.D @ self.M @ self.F.T + 0.5 * self.Lambda, self.H.T @ self.Q @ self.D @ self.mu],
            [(self.H.T @ self.Q @ self.D @ self.M @ self.F.T + 0.5 * self.Lambda).T, self.F @ self.M @ self.F.T, self.F @ self.mu],
            [(self.H.T @ self.Q @ self.D @ self.mu).T, (self.F @ self.mu).T, 1]
        ]) >> 0)
        
        self.constraints.append(cp.bmat([
            [self.M - self.lambda_min * np.eye(self.N_xi), self.mu],
            [self.mu.T, 1]
        ]) >> 0)
        
        self.constraints.append(cp.bmat([
            [self.M - self.N, self.L],
            [self.L.T, self.Sigma_hat]
        ]) >> 0)
        
        self.constraints.append(cp.bmat([
            [self.N, self.mu],
            [self.mu.T, 1]
        ]) >> 0)
        
        self.constraints.append(cp.norm(self.mu_hat, 2) ** 2 - 2 * self.mu.T @ self.mu_hat + cp.trace(self.M + self.Sigma_hat - 2 * self.L) <= self.rho ** 2)
        
        for i in range(self.T):
            for j in range(i + 1):
                row_start = i * self.n_u
                row_end = (i + 1) * self.n_u
                col_start = j * self.n_y
                col_end = (j + 1) * self.n_y
                self.constraints.append(self.Lambda[row_start:row_end, col_start:col_end] == 0)
    
    def define_objective(self):
        self.objective = cp.Maximize(cp.trace(self.D.T @ self.Q @ self.D @ self.M) - cp.trace(cp.inv(self.R + self.H.T @ self.Q @ self.H) @ self.K))
    
    def solve(self):
        prob = cp.Problem(self.objective, self.constraints)
        prob.solve(solver=cp.SCS)
        return prob.value, self.mu.value, self.M.value, self.K.value, self.N.value, self.L.value, self.Lambda.value

if __name__ == "__main__":
    sdp = SDPProblem()
    optimal_value, mu_opt, M_opt, K_opt, N_opt, L_opt, Lambda_opt = sdp.solve()
    print("Optimal value:", optimal_value)
    print("Optimal mu:", mu_opt)
    print("Optimal M:", M_opt)
    print("Optimal K:", K_opt)
    print("Optimal N:", N_opt)
    print("Optimal L:", L_opt)
    print("Optimal Lambda:", Lambda_opt)