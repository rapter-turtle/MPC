import numpy as np

# Define A as 3x3 identity matrix
A = np.eye(3)

# Define Q (an example matrix)
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Solve for P using matrix algebra
P = np.linalg.solve(-A.T - A, -Q)

print("Matrix P:")
print(P)