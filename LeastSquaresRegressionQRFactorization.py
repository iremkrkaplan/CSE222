import numpy as np
from scipy import linalg

x = np.array([1.0, 1.5, 1.9, 2.3, 2.6, 3.0])
y = np.array([2.0, 2.0, 6.0, 7.0, 21.0, 32.0])

X = np.column_stack((np.ones(len(x)), x))
print("Design Matrix X:")
print(X)

Q, R = linalg.qr(X, mode='economic')
print("\nQ matrix:")
print(Q)
print("\nR matrix:")
print(R)

QTy = np.dot(Q.T, y)
beta = linalg.solve(R, QTy)

print("\nQᵀy:")
print(QTy)
print("\nCoefficients (β₀, β₁):")
print(beta)

y_pred = np.dot(X, beta)

y_mean = np.mean(y)
SS_total = np.sum((y - y_mean) ** 2)
SS_residual = np.sum((y - y_pred) ** 2)
r_squared = 1 - (SS_residual / SS_total)

print("\nRegression line: y = {:.4f} + {:.4f}x".format(beta[0], beta[1]))
print("R² value: {:.4f}".format(r_squared))

sse = np.sum((y - y_pred) ** 2)
print("Sum of squared errors: {:.4f}".format(sse))

print("\nPredicted vs Actual values:")
for i in range(len(x)):
    print(f"x = {x[i]}, y_pred = {y_pred[i]:.4f}, y_actual = {y[i]}")