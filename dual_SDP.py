import numpy as np
import scipy as sp
import cvxpy as cp
from LQGSystem import LQGSystem

class SDPProblem:
    def __init__(self, LQG_system):
        self.LQG = LQG_system
        self.define_variables()
        self.define_constraints()
        self.define_objective()
    
    def define_variables(self):
        self.mu = cp.Variable((self.LQG.N_xi, 1))
        self.M = cp.Variable((self.LQG.N_xi, self.LQG.N_xi), PSD=True)
        self.N = cp.Variable((self.LQG.N_xi, self.LQG.N_xi), PSD=True)
        self.L = cp.Variable((self.LQG.N_xi, self.LQG.N_xi))
        self.Lambda = cp.Variable((self.LQG.N_u, self.LQG.N_y))
        self.K = cp.Variable((self.LQG.N_u, self.LQG.N_u), PSD=True)

        # print(f"mu shape: {self.mu.shape}")
        # print(f"M shape: {self.M.shape}")
        # print(f"N shape: {self.N.shape}")
        # print(f"L shape: {self.L.shape}")
        # print(f"Lambda shape: {self.Lambda.shape}")
        # print(f"K shape: {self.K.shape}")
    
    def define_constraints(self):
        self.constraints = []

        # print("K shape:", self.K.shape)
        # print("H.T @ Q @ D @ M @ F.T shape:", (self.H.T @ self.Q @ self.D @ self.M @ self.F.T).shape)
        # print("Lambda shape:", self.Lambda.shape)
        # print("H.T @ Q @ D @ mu shape:", (self.H.T @ self.Q @ self.D @ self.mu).shape)
        # print("F @ M @ F.T shape:", (self.F @ self.M @ self.F.T).shape)
        # print("F @ mu shape:", (self.F @ self.mu).shape)
        # print("constant:", np.array([[1]]).shape)
        
        self.constraints.append(cp.bmat([
            [self.K, self.LQG.H.T @ self.LQG.Q @ self.LQG.D @ self.M @ self.LQG.F.T + 0.5 * self.Lambda, self.LQG.H.T @ self.LQG.Q @ self.LQG.D @ self.mu],
            [(self.LQG.H.T @ self.LQG.Q @ self.LQG.D @ self.M @ self.LQG.F.T + 0.5 * self.Lambda).T, self.LQG.F @ self.M @ self.LQG.F.T, self.LQG.F @ self.mu],
            [(self.LQG.H.T @ self.LQG.Q @ self.LQG.D @ self.mu).T, (self.LQG.F @ self.mu).T, np.array([[1]])]
        ]) >> 0)
        
        self.constraints.append(cp.bmat([
            [self.M - self.LQG.lambda_min * np.eye(self.LQG.N_xi), self.mu],
            [self.mu.T, np.array([[1]])]
        ]) >> 0)
        
        self.constraints.append(cp.bmat([
            [self.M - self.N, self.L],
            [self.L.T, self.LQG.Sigma_hat]
        ]) >> 0)
        
        self.constraints.append(cp.bmat([
            [self.N, self.mu],
            [self.mu.T, np.array([[1]])]
        ]) >> 0)
        
        self.constraints.append(cp.norm(self.LQG.mu_hat, 2) ** 2 - 2 * self.mu.T @ self.LQG.mu_hat + cp.trace(self.M + self.LQG.Sigma_hat - 2 * self.L) <= self.LQG.rho ** 2)
        
        for i in range(self.LQG.T):
            for j in range(i + 1):
                row_start = i * self.LQG.n_u
                row_end = (i + 1) * self.LQG.n_u
                col_start = j * self.LQG.n_y
                col_end = (j + 1) * self.LQG.n_y
                self.constraints.append(self.Lambda[row_start:row_end, col_start:col_end] == 0)
    
    def define_objective(self):
        self.objective = cp.Maximize(cp.trace(self.LQG.D.T @ self.LQG.Q @ self.LQG.D @ self.M) - cp.trace(np.linalg.inv(self.LQG.R + self.LQG.H.T @ self.LQG.Q @ self.LQG.H) @ self.K))
    
    def solve(self):
        prob = cp.Problem(self.objective, self.constraints)
        prob.solve(solver=cp.MOSEK, verbose=False)
        return prob.value, self.mu.value, self.M.value, self.K.value, self.N.value, self.L.value, self.Lambda.value

if __name__ == "__main__":
    optimal_values = []
    optimal_means = []
    optimal_covariances = []
    for T in range(100,101):
        print(T)
        lqg = LQGSystem(n_x=2, n_u=1, n_y=1, T=T)
        sdp = SDPProblem(lqg)
        optimal_value, mu_opt, M_opt, K_opt, N_opt, L_opt, Lambda_opt = sdp.solve()
        # print("Optimal value:", optimal_value)
        # print("Optimal mu:", mu_opt)
        # print("Optimal M:", M_opt)
        # print("Optimal K:", K_opt)
        # print("Optimal N:", N_opt)
        # print("Optimal L:", L_opt)
        # print("Optimal Lambda:", Lambda_opt)
        Sigma_opt = M_opt - mu_opt @ mu_opt.T

        print("mu diff:", np.max(np.abs(mu_opt - sdp.LQG.mu_hat)))
        print("sigma diff:", np.max(np.abs(Sigma_opt - sdp.LQG.Sigma_hat)))# - 2 * sp.linalg.sqrtm(sp.linalg.sqrtm(sdp.Sigma_hat) @ Sigma_opt @ sp.linalg.sqrtm(sdp.Sigma_hat)))) )
        print("squared gelbrich distance:", np.sum(np.linalg.norm(mu_opt - sdp.LQG.mu_hat)**2 + np.linalg.norm(np.trace(Sigma_opt + sdp.LQG.Sigma_hat - 2 * sp.linalg.sqrtm(sp.linalg.sqrtm(sdp.LQG.Sigma_hat) @ Sigma_opt @ sp.linalg.sqrtm(sdp.LQG.Sigma_hat))))) )

        #print(Sigma_opt)
        #print(mu_opt)
        print("Optimal value:", optimal_value)
        optimal_values.append(optimal_value)
        optimal_means.append(mu_opt)
        optimal_covariances.append(Sigma_opt)
    np.save("optimal_values_SDP.npy", optimal_values)
    np.save("optimal_means_SDP.npy", optimal_means)
    np.save("optimal_covariances_SDP.npy", optimal_covariances)