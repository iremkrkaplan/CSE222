import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

print("===== Solving Problem 1(vi) with SymPy =====")
n = sp.symbols('n')
x_n = sp.Function('x')(n)
x_n_plus_1 = sp.Function('x')(n+1)
x_n_plus_2 = sp.Function('x')(n+2)

eq = sp.Eq(x_n_plus_2, 5*x_n_plus_1 + 24*x_n)

print("Setting up the characteristic equation: r^2 - 5r - 24 = 0")
r = sp.symbols('r')
char_eq = r**2 - 5*r - 24
roots = sp.solve(char_eq, r)
print(f"Roots of characteristic equation: {roots}")

print("\nGeneral form of solution: x_n = c1 * 8^n + c2 * (-3)^n")

eq1 = sp.Eq(0, sp.symbols('c1') + sp.symbols('c2'))  # x_0 = c1*8^0 + c2*(-3)^0 = c1 + c2
eq2 = sp.Eq(1, 8*sp.symbols('c1') + (-3)*sp.symbols('c2'))  # x_1 = c1*8^1 + c2*(-3)^1

c1, c2 = sp.symbols('c1 c2')
solution = sp.solve((eq1, eq2), (c1, c2))
print(f"Solving for constants with initial conditions x_0 = 0, x_1 = 1:")
print(f"c1 = {solution[c1]}, c2 = {solution[c2]}")

x_n_formula = solution[c1] * roots[0]**n + solution[c2] * roots[1]**n
print(f"\nFinal formula for x_n: {x_n_formula}")
print(f"Simplified: x_n = (1/11) * (8^n - (-3)^n)")

print("\nFirst 10 terms of the sequence:")
for i in range(10):
    term_value = float(x_n_formula.subs(n, i))
    print(f"x_{i} = {term_value}")

print("\n===== Solving Problem 3(vi) with SymPy - Lagrange Interpolation =====")
points = [(-2, -14), (-1, -3.5), (1, 2.5), (2, 10)]

x = sp.symbols('x')

L = []
for i in range(len(points)):
    numerator = 1
    denominator = 1
    for j in range(len(points)):
        if i != j:
            numerator *= (x - points[j][0])
            denominator *= (points[i][0] - points[j][0])
    L.append(numerator / denominator)

p = sum(points[i][1] * L[i] for i in range(len(points)))

p_expanded = sp.expand(p)
print(f"Lagrange interpolation polynomial: {p_expanded}")

coeffs = sp.Poly(p_expanded, x).all_coeffs()
degrees = range(len(coeffs)-1, -1, -1)
print("\nCoefficients as fractions:")
for degree, coeff in zip(degrees, coeffs):
    print(f"Coefficient of x^{degree}: {sp.Rational(coeff)}")

def poly_func(x_val):
    return float(p.subs(x, x_val))

print("\nVerification at given points:")
for point in points:
    x_val, y_expected = point
    y_calculated = poly_func(x_val)
    print(f"At x = {x_val}: p(x) = {y_calculated}, Expected: {y_expected}")

x_vals = np.linspace(-3, 3, 100)
y_vals = [poly_func(x_val) for x_val in x_vals]

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, 'b-', label='Interpolation Polynomial')
plt.scatter([p[0] for p in points], [p[1] for p in points], color='red', s=50, label='Data Points')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Lagrange Interpolation Polynomial')
plt.grid(True)
plt.legend()
plt.savefig('sympy_interpolation_plot.png')
plt.close()

print("Plot saved as 'sympy_interpolation_plot.png'")