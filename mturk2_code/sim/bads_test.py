import numpy as np
from pybads.bads import BADS

# 1. Define the black-box function to minimize
def objective(x):
    # x is a NumPy array of shape (2,)
    return np.sin(x[0]) + np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2)

# 2. Set a starting point for optimization
x0 = np.array([2.0, 2.0])  # Initial guess

# 3. Define parameter bounds
# Each bound is a tuple: (lower, upper)
lb = np.array([-5.0, -5.0])
ub  = np.array([5.0, 5.0])

# 4. Create and run the BADS optimizer
bads = BADS(fun=objective, x0=x0, lower_bounds=lb, upper_bounds=ub)
res = bads.optimize()

# 5. Print result
print(f"Optimal parameters: {res.x}")
print(f"Minimum value: {res.fval}")
