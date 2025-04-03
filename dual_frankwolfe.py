import numpy as np
import torch

class FrankWolfeOptimizer:
    def __init__(self, objective_func, gradient_func, constraint_func, initial_point, max_iter=1000, tol=1e-6):
        """
        Initializes the Frank-Wolfe optimizer.
        
        :param objective_func: Callable, the convex objective function to minimize.
        :param gradient_func: Callable, the gradient of the objective function.
        :param constraint_func: Callable, the projection operator onto the feasible set.
        :param initial_point: np.ndarray, the initial point in the feasible set.
        :param max_iter: int, maximum number of iterations.
        :param tol: float, the tolerance for stopping criterion.
        """
        self.objective_func = objective_func
        self.gradient_func = gradient_func
        self.constraint_func = constraint_func
        self.x = initial_point
        self.max_iter = max_iter
        self.tol = tol
    
    def optimize(self):
        """
        Perform the Frank-Wolfe optimization.
        
        :return: The optimal point found and the corresponding objective value.
        """
        for k in range(self.max_iter):
            # Step 1: Compute the gradient at the current point
            grad = self.gradient_func(self.x)
            
            # Step 2: Find the minimizer of the linear approximation (the "Frank-Wolfe direction")
            s = self.linear_minimization(grad)
            
            # Step 3: Compute the step size (using a fixed step size or line search)
            gamma = self.line_search(s, k)
            
            # Step 4: Update the current point
            new_x = (1 - gamma) * self.x + gamma * s
            
            # Step 5: Check for convergence
            if np.linalg.norm(new_x - self.x) < self.tol:
                print(f"Converged at iteration {k}.")
                return new_x, self.objective_func(new_x)
            
            self.x = new_x
        
        print(f"Max iterations reached.")
        return self.x, self.objective_func(self.x)
    
    def linear_minimization(self, grad):
        """
        Solve the linear minimization problem (find s that minimizes the linear approximation).
        
        :param grad: np.ndarray, the gradient at the current point.
        :return: s, the point that minimizes the linear approximation.
        """
        # Assuming the feasible region is a polytope (for example, a simple box constraint).
        s = self.constraint_func(grad)
        return s
    
    def line_search(self, s, k):
        """
        Perform a line search to find the optimal step size gamma.
        
        :param s: np.ndarray, the candidate direction (Frank-Wolfe direction).
        :return: gamma, the step size.
        """
        # Here, we use a simple fixed step size; this could be replaced with a backtracking line search.
        gamma = 2 / (k + 2)  # Common step size rule for Frank-Wolfe
        return gamma


# Example Usage:

# Define a convex quadratic objective function: f(x) = 0.5 * x^T A x - b^T x
A = np.array([[3, 1], [1, 2]])
b = np.array([1, 1])
objective_func = lambda x: 0.5 * np.dot(x, A @ x) - np.dot(b, x)
gradient_func = lambda x: A @ x - b

# Constraint set: Assume the feasible region is the unit ball
constraint_func = lambda grad: np.clip(grad, -1, 1)  # Projection onto [-1, 1] box

# Initial point
x0 = np.array([0.5, 0.5])

# Instantiate and run the optimizer
optimizer = FrankWolfeOptimizer(objective_func, gradient_func, constraint_func, x0)
optimal_point, optimal_value = optimizer.optimize()

# Create a tensor with requires_grad=True to enable automatic differentiation
x = torch.tensor(2.0, requires_grad=True)

# Define a simple function
y = x**2 + 3*x + 2

# Compute the gradient (derivative) of y with respect to x
y.backward()

# Access the gradient of x
print(x.grad)

# Output the result
print(f"Optimal point: {optimal_point}")
print(f"Optimal value: {optimal_value}")
