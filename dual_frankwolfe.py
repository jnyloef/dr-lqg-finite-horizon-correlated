import numpy as np
import torch
from LQGSystem import LQGSystem
import matplotlib.pyplot as plt

class FrankWolfeOptimizer:
    def __init__(self, LQG_system, max_iter=100, delta=0.9, tol=1e-6):
        """
        Initializes the Frank-Wolfe optimizer for optimizing over A and Sigma.

        :param objective_func: Callable, the objective function to minimize.
        :param A: torch.Tensor, initial point for A.
        :param Sigma: torch.Tensor, initial point for Sigma.
        :param max_iter: int, maximum number of iterations.
        :param tol: float, tolerance for stopping criterion.
        :param learning_rate: float, step size (fixed in this case).
        """
        self.LQG = LQG_system

        # initialize
        self.mu = torch.tensor(self.LQG.mu_hat, requires_grad=True)
        self.Sigma = torch.tensor(self.LQG.Sigma_hat, requires_grad=True)

        self.max_iter = max_iter
        self.delta = delta
        self.tol = tol

    def compute_cost(self):
        """Computes the objective function f(mu, Sigma) for the SDP."""
        
        # Extract required matrices from self.LQG
        H, Q, D, F = torch.tensor(self.LQG.H), torch.tensor(self.LQG.Q), torch.tensor(self.LQG.D), torch.tensor(self.LQG.F)
        R = torch.tensor(self.LQG.R)  # Control cost matrix
        
        T, n_u, n_y = self.LQG.T, self.LQG.n_u, self.LQG.n_y  # Time horizon and dimensions
        
        # Compute J using the Kronecker product
        self.J = torch.kron(F @ self.Sigma @ F.T, R + H.T @ Q @ H)
        
        # Compute C = vec(Hᵀ Q D Σ* Fᵀ)
        C = (H.T @ Q @ D @ self.Sigma @ F.T).T.flatten()  # Column-major flattening (row wise flattening by default, hence .T)
        
        # Construct Z
        num_elements = T * n_u * T * n_y  # Total elements in vec(U)
        indices = []  # To store indices corresponding to non-lower-triangular elements
        for i in range(T):  # Block row index
            for j in range(i + 1, T):  # Block column index (strictly upper triangular)
                for row in range(n_u):  # Rows within block
                    for col in range(n_y):  # Columns within block
                        index = (i * n_u + row) + (j * n_y + col) * (T * n_u)
                        indices.append(index)
                        
        Z = np.zeros((len(indices), num_elements))  # Selection matrix
        for new_idx, old_idx in enumerate(indices):
            Z[new_idx, old_idx] = 1  # Pick out strictly upper triangular elements
        self.Z = torch.tensor(Z)  # Convert to tensor

        # Compute the inverse of (Z J⁻¹ Zᵀ)
        self.J_inv = torch.kron(torch.linalg.inv(F @ self.Sigma @ F.T), torch.linalg.inv(R + H.T @ Q @ H))
        print("norm of J_inv:", torch.norm(self.J_inv).item())
        self.ZJZ_inv = torch.linalg.inv(self.Z @ self.J_inv @ self.Z.T)
        print("norm of ZJZ_inv:", torch.norm(self.ZJZ_inv).item())
        
        # First term: Cᵀ (J⁻¹ Zᵀ (Z J⁻¹ Zᵀ)⁻¹ Z J⁻¹ - J⁻¹) C
        term1 = C.T @ (self.J_inv @ self.Z.T @ self.ZJZ_inv @ self.Z @ self.J_inv - self.J_inv) @ C
        print("term 1:", term1.item())
        
        # Second term: Tr(Dᵀ Q D Σ*)
        term2 = torch.trace(D.T @ Q @ D @ self.Sigma)
        print("term 2:", term2.item())
        
        # Third term: μ*ᵀ Dᵀ (Q - QH(R + Hᵀ Q H)⁻¹ Hᵀ Q) D μ*
        term3 = self.mu.T @ D.T @ (Q - Q @ H @ torch.linalg.inv(R + H.T @ Q @ H) @ H.T @ Q) @ D @ self.mu
        print("term 3:", term3.item())
        
        # Final cost function value
        cost = term1 + term2 + term3
        
        return cost


    def minimization_oracle(self):
        """
        Compute the minimization oracle for the Frank-Wolfe algorithm.
        
        :return: The optimal mu and Sigma for the linearization oracle.
        """
        Sigma_hat = torch.tensor(self.LQG.Sigma_hat)
        mu_hat = torch.tensor(self.LQG.mu_hat)

        # recompute cost, a function of self.mu and self.Sigma
        self.objective_func = self.compute_cost()
        print("objective:", self.objective_func)

        # Compute the gradient of the objective function
        grad_mu = torch.autograd.grad(self.objective_func, self.mu)[0]
        grad_Sigma_non_sym = torch.autograd.grad(self.objective_func, self.Sigma)[0]
        grad_Sigma = 0.5 * (grad_Sigma_non_sym + grad_Sigma_non_sym.T) # CHECK THIS!

        print("norm of gradient:", torch.norm(grad_Sigma_non_sym).item())
        print("smallest eigenval of objective gradient:", np.linalg.eigh(grad_Sigma)[0][0].item())
        #input("OK?:")
        assert np.linalg.eigh(grad_Sigma)[0][0] >= - 1e-6 # TOL smallest eigenvalue (sorted in ascending order)


        lambda_1 = torch.linalg.eigh(grad_Sigma)[0][-1] # largest eigenvalue (ascending order)
        v_1 = torch.linalg.eigh(grad_Sigma)[1][:, -1] # corresponding eigenvector

        # Compute the linearization oracle via bisection
        gamma_low = torch.max(torch.tensor(0), lambda_1)
        gamma_high = torch.max(torch.norm(grad_mu)**2/(self.LQG.rho * np.sqrt(2)), lambda_1 * (1 + torch.sqrt(2 * torch.trace(Sigma_hat)) / self.LQG.rho)) # 0.5 * lambda_1 * (1 + torch.sqrt(torch.trace(Sigma_hat)) / self.LQG.rho) # fix tomorrow so tht mu is there! use your formula instead
        #100*lambda_1 * (1 + torch.sqrt(torch.trace(Sigma_hat)) / self.LQG.rho) # fix tomorrow so tht mu is there! use your formula instead
        #gamma_low = lambda_1 * (1 + torch.sqrt(v_1 @ Sigma_hat @ v_1) / self.LQG.rho) # 0.5 * lambda_1 * (1 + torch.sqrt(torch.trace(Sigma_hat)) / self.LQG.rho) # fix tomorrow so tht mu is there! use your formula instead
        #gamma_high = lambda_1 * (1 + torch.sqrt(torch.trace(Sigma_hat)) / self.LQG.rho) # fix tomorrow so tht mu is there! use your formula instead

        phi = lambda gamma: gamma**2 * torch.trace(torch.linalg.inv(gamma * torch.eye(self.LQG.N_xi) - grad_Sigma) @ Sigma_hat) + gamma * (self.LQG.rho**2 - torch.trace(Sigma_hat)) + torch.linalg.norm(grad_mu,2)**2 / (4 * gamma) + grad_mu.T @ mu_hat - torch.trace(grad_Sigma @ self.Sigma) - grad_mu.T @ self.mu
        dphi = lambda gamma: self.LQG.rho**2 - torch.trace( Sigma_hat @ torch.linalg.matrix_power(torch.eye(self.LQG.N_xi) - gamma * torch.linalg.inv(gamma * torch.eye(self.LQG.N_xi) - grad_Sigma) , 2) ) - torch.linalg.norm(grad_mu,2)**2 / (4 * gamma**2)

        # Bisection method to find the optimal gamma
        iter = 0
        while True:
            iter += 1
            gamma_mid = (gamma_low + gamma_high) / 2
            D_gamma = torch.linalg.inv(gamma_mid * torch.eye(self.LQG.N_xi) - grad_Sigma)
            new_mu = grad_mu / (2 * gamma_mid) + mu_hat
            new_Sigma = gamma_mid**2 * D_gamma @ Sigma_hat @ D_gamma
            #print(f"Iteration {iter}: dphi(gamma_mid) = {dphi(gamma_mid)}, phi(gamma_mid) = {phi(gamma_mid)}, trace = {torch.trace(grad_Sigma @ (new_Sigma - self.Sigma)) + grad_mu.T @ (new_mu - self.mu)}")
            if dphi(gamma_mid) > 0 and torch.trace(grad_Sigma @ (new_Sigma - self.Sigma)) + grad_mu.T @ (new_mu - self.mu) >= self.delta * phi(gamma_mid):
                break
            if dphi(gamma_mid) < 0:
                gamma_low = gamma_mid
            else:
                gamma_high = gamma_mid
            print(f"Iteration {iter}: dphi(gamma_mid) = {dphi(gamma_mid).item()}, gamma_low = {gamma_low.item()}, gamma_high = {gamma_high.item()}")
            #input()
        return new_mu, new_Sigma


    def optimize(self):
        """
        Perform the Frank-Wolfe optimization.
        
        :return: The optimal values of A and Sigma, and the corresponding objective value.
        """
        for k in range(self.max_iter):
            # Zero gradients from the previous step (although we don't need them here)
            if self.mu.grad is not None:
                self.mu.grad.zero_()
            if self.Sigma.grad is not None:
                self.Sigma.grad.zero_()

            # Step 1: Compute minimization oracle
            [new_mu, new_Sigma] = self.minimization_oracle()
            
            # Step 2: Update mu and Sigma using the Frank-Wolfe direction
            Sigma_old = self.Sigma.clone()
            mu_old = self.mu.clone()
            learning_rate = 2/(k+2) # Frank-Wolfe step size
            
            self.mu = (self.mu + learning_rate * (new_mu - self.mu)).requires_grad_(True)
            self.Sigma = (self.Sigma + learning_rate * (new_Sigma - self.Sigma)).requires_grad_(True)

            # Step 3: Check for convergence (e.g., change in parameters or objective value)
            if torch.sqrt(torch.norm(self.Sigma - Sigma_old)**2 + torch.norm(self.mu - mu_old)**2) < self.tol:
                print(f"Converged at iteration {k}.")
                return self.objective_func.item(), np.array(self.mu.detach()), np.array(self.Sigma.detach())
            
            print(f"Iteration {k}: Objective value = {self.objective_func.item()}, mu norm = {torch.norm(self.mu).item()}, Sigma norm = {torch.norm(self.Sigma).item()}")
        
        print("Max iterations reached.")
        return self.objective_func.item(), np.array(self.mu.detach()), np.array(self.Sigma.detach())
    
if __name__ == "__main__":
    optimal_values = []
    optimal_means = []
    optimal_covariances = []
    # unit test
    for T in range(8,25):
        lqg = LQGSystem(n_x=2, n_u=2, n_y=2, T=T)
        optimizer = FrankWolfeOptimizer(lqg, max_iter=100, delta=1 - 1e-5, tol=1e-3)
        
        # cost function test
        optimizer.compute_cost()
        U_test = np.random.randn(lqg.N_u, lqg.N_y)
        Z_test = np.array(optimizer.Z)
        flat_U_test = U_test.T.flatten()
        if Z_test.shape[0] > 0:
            # plt.spy(Z_test.T @ Z_test)
            # plt.show()
            assert np.linalg.matrix_rank(Z_test) == Z_test.shape[0]
            column_indices = np.nonzero(Z_test)[1]
            complem_column_indices = np.setdiff1d(np.arange(lqg.N_u * lqg.N_y), column_indices)
            #print(column_indices)
            #print(np.sum(Z_test, axis=1))
            flat_U_test[column_indices] = 0
        U_strict_u_tril = flat_U_test.reshape((lqg.N_y, lqg.N_u)).T
        #plt.spy(U_strict_u_tril)
        #plt.show()



        # Run the optimization
        mu_optimal, Sigma_optimal, optimal_value = optimizer.optimize()

        # Output the result
        #print(f"Optimal mu:\n{mu_optimal}")
        #print(f"Optimal Sigma:\n{Sigma_optimal}")
        print(f"Optimal value: {optimal_value}")
        optimal_values.append(optimal_value)
        optimal_means.append(mu_optimal)
        optimal_covariances.append(Sigma_optimal)
        # Save the results
    #np.save("optimal_values_fw.npy", optimal_values)
    #np.save("optimal_means_fw.npy", optimal_means)
    #np.save("optimal_covariances_fw.npy", optimal_covariances)
