import numpy as np
from numpy.linalg import eig, matrix_power
from sympy import symbols, expand
import matplotlib.pyplot as plt
from fractions import Fraction

print("===== Problem 1(vi) - Detailed Matrix Calculations =====")
A = np.array([[5, 24], [1, 0]])
print(f"Matrix A:\n{A}")

eigenvalues, eigenvectors = eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

v1 = np.array([8, 1])
Av1 = A @ v1
print(f"\nVerifying eigenvector for λ = 8:")
print(f"A·v1 = {Av1}")
print(f"λ·v1 = {8 * v1}")

v2 = np.array([-3, 1])
Av2 = A @ v2
print(f"\nVerifying eigenvector for λ = -3:")
print(f"A·v2 = {Av2}")
print(f"λ·v2 = {-3 * v2}")

x0 = 0
x1 = 1
init_vector = np.array([x1, x0])

V = np.column_stack((v1, v2))
print(f"\nEigenvector matrix V:\n{V}")

c = np.linalg.solve(V, init_vector)
print(f"Constants c1, c2: {c}")
print(f"As fractions: c1 = {Fraction(c[0]).limit_denominator()}, c2 = {Fraction(c[1]).limit_denominator()}")

print("\nCalculating the first 10 terms of the sequence:")
print("n | Recurrence | Formula")
print("-" * 30)

for n in range(10):
    if n == 0:
        recurrence_term = x0
    elif n == 1:
        recurrence_term = x1
    else:
        recurrence_term = 5 * prev_term + 24 * prev_prev_term

    formula_term = c[0] * 8 ** n + c[1] * (-3) ** n

    prev_prev_term = prev_term if n > 0 else x0
    prev_term = recurrence_term if n > 0 else x1

    print(f"{n} | {recurrence_term:10.1f} | {formula_term:10.6f}")

print("\n===== Problem 2(vi) - Predator-Prey Analysis =====")
A = np.array([[0.4, 0.5], [-0.4, 1.3]])
initial = np.array([60, 50])
print(f"Matrix A:\n{A}")
print(f"Initial population [x0, y0]: {initial}")

eigenvalues, eigenvectors = eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Magnitudes of eigenvalues: {np.abs(eigenvalues)}")

I = np.eye(2)
print(f"\nMatrix (I-A):\n{I - A}")
print(f"Determinant of (I-A): {np.linalg.det(I - A)}")

steps = 50
populations = np.zeros((steps, 2))
populations[0] = initial

print("\nPopulation dynamics over time:")
print("Step |   Predator   |    Prey     |    Ratio    ")
print("-" * 45)

for i in range(1, steps):
    populations[i] = A @ populations[i - 1]
    ratio = populations[i, 0] / populations[i, 1] if populations[i, 1] != 0 else float('inf')

    if i < 5 or i > steps - 5:  # Show first and last few steps
        print(f"{i:4d} | {populations[i, 0]:11.6f} | {populations[i, 1]:11.6f} | {ratio:11.6f}")
    elif i == 5:
        print("...")

print("\n===== Problem 3(vi) - Lagrange Interpolation Calculations =====")
points = [(-2, -14), (-1, -3.5), (1, 2.5), (2, 10)]
x_points, y_points = zip(*points)
print(f"Points: {points}")

x = symbols('x')

print("\nLagrange basis polynomials:")
L_basis = []
for i in range(len(points)):
    numerator = 1
    denominator = 1
    for j in range(len(points)):
        if i != j:
            numerator *= (x - x_points[j])
            denominator *= (x_points[i] - x_points[j])
    L_i = numerator / denominator
    L_basis.append(L_i)
    print(f"L_{i}(x) = {expand(L_i)}")

p = sum(y_points[i] * L_basis[i] for i in range(len(points)))
p_expanded = expand(p)
print(f"\nInterpolation polynomial: p(x) = {p_expanded}")

print("\nCoefficients as fractions:")
coeffs = p_expanded.as_poly().all_coeffs()
degrees = range(len(coeffs) - 1, -1, -1)  # Degrees from highest to lowest
for degree, coeff in zip(degrees, coeffs):
    frac = Fraction(float(coeff)).limit_denominator()
    print(f"Coefficient of x^{degree}: {frac}")

print("\nVerification at given points:")
for point in points:
    x_val, y_expected = point
    y_calculated = float(p.subs(x, x_val))
    print(f"At x = {x_val}: p(x) = {y_calculated}, Expected: {y_expected}")