import numpy as np

np.random.seed(0)

class LQGSystem:
    def __init__(self, n_x, n_y, n_u, T):
            self.n_x = n_x
            self.n_y = n_y
            self.n_u = n_u
            self.T = T
            
            self.N_x = (T+1) * n_x
            self.N_u = T * n_u
            self.N_y = T * n_y
            self.N_xi = n_x + T * (n_x + n_y)
            
            self.initialize_matrices()

    def initialize_matrices(self):
        # Initialize matrices to something we want here.
        self.A_sys = np.diag(np.random.uniform(0, 1, self.n_x))
        self.B_sys = np.random.randn(self.n_x, self.n_u)
        self.C_sys = np.random.randn(self.n_y, self.n_x)

        self.D_sys = np.hstack( [np.eye(self.n_x) , np.zeros( (self.n_x, self.n_y) )] )
        self.E_sys = np.hstack( [np.zeros( (self.n_y, self.n_x) ) , np.eye(self.n_y)] )
        
        self.H = np.zeros((self.N_x, self.N_u))
        for t in range(1, self.T + 1):
            for k in range(t):
                self.H[t * self.n_x:(t + 1) * self.n_x, k * self.n_u:(k + 1) * self.n_u] = np.linalg.matrix_power(self.A_sys, t - 1 - k) @ self.B_sys
        
        self.D = np.zeros((self.N_x, self.N_xi))
        for t in range(0, self.T + 1):
            for k in range(t + 1):
                block = np.linalg.matrix_power(self.A_sys, t - k)
                if k == 0:
                    self.D[t * self.n_x : (t + 1) * self.n_x, k * self.n_x : (k + 1) * self.n_x] = block
                else:
                    block = block @ self.D_sys
                    self.D[t * self.n_x : (t + 1) * self.n_x, k * (self.n_x + self.n_y) - self.n_y : (k + 1) * (self.n_x + self.n_y) - self.n_y] = block


        self.C = np.kron( np.hstack([np.eye(self.T), np.zeros((self.T,1))]) , self.C_sys)
        self.E = np.hstack( [ np.zeros((self.N_y, self.n_x)), np.kron(self.E_sys , np.eye(self.T)) ] ) # has dim N_y x N_xi
        
        self.F = self.C @ self.D + self.E
        
        A_Q = np.random.randn(self.N_x, self.N_x)
        self.Q = (A_Q.T @ A_Q)/self.N_x # positive semidefinite
        
        B_R = np.random.randn(self.N_u, self.N_u)
        self.R = (B_R.T @ B_R + np.eye(self.N_u))/self.N_u # positive definite
        
        C_Sigma = np.random.randn(self.N_xi, self.N_xi)
        self.Sigma_hat = (C_Sigma.T @ C_Sigma + np.eye(self.N_xi))/self.N_xi # positive deifnite center covariance
        print("Term 2:", np.linalg.norm(self.D.T @ self.Q @ self.D @ self.Sigma_hat))
        
        self.lambda_min = np.min(np.linalg.eigvals(self.Sigma_hat)) # smallest eigenvalue of Sigma_hat
        self.rho = 1.0 # radius
        self.mu_hat = 100*np.random.randn(self.N_xi, 1) # any center mean

        print(f"H shape: {self.H.shape}")
        print(f"D shape: {self.D.shape}")
        print(f"C shape: {self.C.shape}")
        print(f"E shape: {self.E.shape}")
        print(f"F shape: {self.F.shape}")
        print(f"Q shape: {self.Q.shape}")
        print(f"R shape: {self.R.shape}")
        print(f"Sigma_hat shape: {self.Sigma_hat.shape}")
        print(f"mu_hat shape: {self.mu_hat.shape}")