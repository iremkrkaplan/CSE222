import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, expand
from numpy.linalg import eig, matrix_power

def problem1_vi():
    print("\n--- Problem 1(vi): Recursive Sequence ---")
    A = np.array([[5, 24], [1, 0]])
    print("Matrix A:\n", A)

    eigenvalues, eigenvectors = eig(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    x0 = 0
    x1 = 1
    init_vector = np.array([x1, x0])

    c = np.linalg.solve(eigenvectors, init_vector)
    print("Constants c1, c2:", c)

    terms = []
    for n in range(10):
        if n == 0:
            terms.append(x0)
        elif n == 1:
            terms.append(x1)
        else:
            terms.append(5 * terms[n - 1] + 24 * terms[n - 2])

    print("First 10 terms of the sequence:", terms)

    formula_terms = []
    for n in range(10):
        term = c[0] * eigenvalues[0] ** n * eigenvectors[0, 0] + c[1] * eigenvalues[1] ** n * eigenvectors[0, 1]
        formula_terms.append(
            term.real)

    print("Formula-derived terms (verification):", [round(t, 10) for t in formula_terms])

    lambda1, lambda2 = eigenvalues
    print(f"General term: x_n = {c[0]:.6f} * ({lambda1:.6f})^n - {-c[1]:.6f} * ({lambda2:.6f})^n")
    print(f"Simplified: x_n = (1/11) * (8^n - (-3)^n)")


def problem2_vi():
    print("\n--- Problem 2(vi): Predator-Prey Model ---")
    A = np.array([[0.4, 0.5], [-0.4, 1.3]])
    initial = np.array([60, 50])

    print("Matrix A:\n", A)
    print("Initial population [x0, y0]:", initial)

    eigenvalues, _ = eig(A)
    print("Eigenvalues:", eigenvalues)


    I = np.eye(2)
    if abs(np.linalg.det(I - A)) < 1e-10:
        print("System has a non-trivial fixed point")
    else:
        print("Fixed point is (0, 0)")

    steps = 50
    populations = np.zeros((steps, 2))
    populations[0] = initial

    for i in range(1, steps):
        populations[i] = A @ populations[i - 1]

    ratios = populations[:, 0] / np.where(populations[:, 1] == 0, 1, populations[:, 1])  # Avoid division by zero

    print(f"After {steps - 1} steps:")
    print(f"Predator population (x): {populations[-1, 0]:.6f}")
    print(f"Prey population (y): {populations[-1, 1]:.6f}")
    print(f"Ratio x/y: {ratios[-1]:.6f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(steps), populations[:, 0], 'r-', label='Predator (x)')
    plt.plot(range(steps), populations[:, 1], 'b-', label='Prey (y)')
    plt.xlabel('Time (months)')
    plt.ylabel('Population')
    plt.title('Population over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(populations[:, 1], populations[:, 0], 'g-')
    plt.scatter(populations[0, 1], populations[0, 0], color='red', s=50, label='Initial')
    plt.xlabel('Prey (y)')
    plt.ylabel('Predator (x)')
    plt.title('Phase Diagram')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('predator_prey_plot.png')
    plt.close()

    print("Plot saved as 'predator_prey_plot.png'")


def problem3_vi():
    print("\n--- Problem 3(vi): Lagrange Interpolation ---")
    # Given points
    points = [(-2, -14), (-1, -3.5), (1, 2.5), (2, 10)]
    x_points, y_points = zip(*points)

    print("Points:", points)

    x = symbols('x')

    L = []
    for i in range(len(points)):
        basis = 1
        for j in range(len(points)):
            if i != j:
                basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
        L.append(basis)

    p = 0
    for i in range(len(points)):
        p += y_points[i] * L[i]

    p_expanded = expand(p)
    print("Interpolation polynomial:")
    print(p_expanded)

    def poly_func(x_val):
        return float(p.subs(x, x_val))

    for point in points:
        x_val, y_expected = point
        y_calculated = poly_func(x_val)
        print(f"At x = {x_val}: p(x) = {y_calculated:.6f}, Expected: {y_expected}")

    x_vals = np.linspace(-3, 3, 100)
    y_vals = [poly_func(x_val) for x_val in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, 'b-', label='Interpolation Polynomial')
    plt.scatter(x_points, y_points, color='red', s=50, label='Data Points')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Lagrange Interpolation Polynomial')
    plt.grid(True)
    plt.legend()
    plt.savefig('interpolation_plot.png')
    plt.close()

    print("Plot saved as 'interpolation_plot.png'")


def main():
    print("MAT 222 Linear Algebra - Assignment 1 Solutions")
    print("Solution of 6th parts")

    problem1_vi()
    problem2_vi()
    problem3_vi()

if __name__ == "__main__":
    main()