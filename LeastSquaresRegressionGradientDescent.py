import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 1.5, 1.9, 2.3, 2.6, 3.0])
y = np.array([2.0, 2.0, 6.0, 7.0, 21.0, 32.0])
X = np.column_stack((np.ones(len(x)), x))

def compute_cost(X, y, beta):
    m = len(y)
    predictions = X.dot(beta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, beta, learning_rate, epsilon):
    m = len(y)
    cost_history = []
    gradient_norm_history = []
    beta_history = [beta.copy()]
    iteration = 0

    while True:
        predictions = X.dot(beta)

        gradients = (1 / m) * X.T.dot(predictions - y)

        gradient_norm = np.linalg.norm(gradients)
        gradient_norm_history.append(gradient_norm)

        if gradient_norm < epsilon:
            break

        beta = beta - learning_rate * gradients

        beta_history.append(beta.copy())
        cost = compute_cost(X, y, beta)
        cost_history.append(cost)

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}, Gradient Norm: {gradient_norm:.6f}")

        iteration += 1

    return beta, cost_history, gradient_norm_history, beta_history, iteration


initial_beta = np.array([0.0, 0.0])  # starting with (0, 0)
learning_rate = 0.001
epsilon = 0.005

final_beta, cost_history, gradient_norm_history, beta_history, iterations = gradient_descent(X, y, initial_beta,
                                                                                             learning_rate, epsilon)

print("\nGradient Descent Results:")
print(f"Final coefficients (β₀, β₁): {final_beta}")
print(f"Final cost: {compute_cost(X, y, final_beta):.6f}")
print(f"Final gradient norm: {gradient_norm_history[-1]:.6f}")
print(f"Iterations required: {iterations}")
print(f"Regression line: y = {final_beta[0]:.4f} + {final_beta[1]:.4f}x")

y_pred = X.dot(final_beta)


y_mean = np.mean(y)
SS_total = np.sum((y - y_mean) ** 2)
SS_residual = np.sum((y - y_pred) ** 2)
r_squared = 1 - (SS_residual / SS_total)
print(f"R² value: {r_squared:.4f}")

plt.figure(figsize=(10, 6))

plt.scatter(x, y, color='blue', label='Data points')

x_line = np.linspace(min(x), max(x), 100)
y_line = final_beta[0] + final_beta[1] * x_line
plt.plot(x_line, y_line, color='red', label=f'Regression line: y = {final_beta[0]:.4f} + {final_beta[1]:.4f}x')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Regression Line using Gradient Descent')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History during Gradient Descent')
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(range(len(gradient_norm_history)), gradient_norm_history)
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm History during Gradient Descent')
plt.grid(True)
plt.axhline(y=epsilon, color='r', linestyle='--', label=f'Threshold ε={epsilon}')
plt.legend()

print("\nPredicted vs Actual values:")
for i in range(len(x)):
    print(f"x = {x[i]}, y_pred = {y_pred[i]:.4f}, y_actual = {y[i]}")

plt.tight_layout()
plt.show()