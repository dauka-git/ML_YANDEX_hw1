import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(0)

def rref(A, tol=1e-6):

    R = np.array(A, dtype=float)
    m,n = R.shape
    pivots=[]
    r=0

    for c in range(n):
        if r==m: break
        
        pivot_row=r
        for i in range(r+1, m):
            if abs(R[i,c]) > abs(R[pivot_row, c]):
                pivot_row=i
        if abs(R[pivot_row, c]) < tol: continue
        
        R[[r, pivot_row], :] = R[[pivot_row, r], : ]
        R[r, :] = R[r, :] / R[r,c]
        for i in range(m): 
            if i != r:
                R[i, :] = R[i, :] - R[i, c] * R[r, :]
                R[i,c] = 0.0
        pivots.append(c)
        r+=1
        
    return R, pivots

def dual_basis(P):
    P = np.array(P, dtype=float)
    n = P.shape[0]

    aug = np.zeros((n, 2*n))
    aug[:, :n] = P
    for i in range(n):
        aug[i, n+i] = 1.0
    R, pivots = rref(aug)
    if pivots != list(range(n)):
        raise ValueError("P must be invertible")
    return R[:, n:]

def lag_coef(nodes):
    n = len(nodes)
    V= np.zeros((n,n))
    for i in range(n):
        V[i, 0]=1.0
        for j in range(1,n):
            V[i,j] = V[i, j-1] * nodes[i]
    return dual_basis(V)

def polyn_val(coef, t):
    val = 0.0
    for j in range(len(coef)-1, -1, -1):
        val = val*t + coef[j]
    return val

def interpol(nodes, vals):
    L = lag_coef(nodes)
    coef = np.zeros(len(nodes))
    for k in range(len(nodes)):
        coef = coef + vals[k] * L[:, k]
    return coef


checks =[ ]
P = rng.standard_normal((5, 5))
rows = dual_basis(P)
checks.append(("Dual basis: r_i b_j = delta_ij",
                np.allclose(rows @ P, np.eye(5))))
checks.append(("Dual basis agrees with NumPy inverse",
                np.allclose(rows, np.linalg.inv(P))))
x = rng.standard_normal(5)
coordinates = rows @ x
reconstructed = np.zeros(5)
for i in range(5):
    reconstructed = reconstructed + coordinates[i] * P[:, i]
checks.append(("Coordinates reconstruct x", np.allclose(reconstructed, x)))

nodes = np.array([-1.0, 0.0, 1.0])
L = lag_coef(nodes)
expected = np.array([[0, 1, 0], [-0.5, 0, 0.5], [0.5, -1, 0.5]])
checks.append(("Lagrange coefficients match Example 8.9", np.allclose(L, expected)))
evaluations = np.zeros((3, 3))
for k in range(3):
    for j in range(3):
        evaluations[k, j] = polyn_val(L[:, j], nodes[k])
checks.append(("L_j(t_k) = delta_jk", np.allclose(evaluations, np.eye(3))))
print("Lagrange coefficient columns for nodes -1, 0, 1 (powers 1, t, t^2):")
print(L)
print("L_0(t) = (t^2 - t)/2; L_1(t) = 1 - t^2; L_2(t) = (t^2 + t)/2")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
cases = [
    ("Sine: 9 equally spaced nodes", -1, 1, 9, lambda t: np.sin(3 * t)),
    ("Runge: 21 equally spaced nodes", -5, 5, 21, lambda t: 1 / (1 + t * t)),
]
for ax, (title, a, b, n, f) in zip(axes, cases):
    nodes = np.linspace(a, b, n)
    values = f(nodes)
    coefficients = interpol(nodes, values)
    
    checks.append((title + ": interpolation at nodes",
                    np.allclose(polyn_val(coefficients, nodes), values,
                                rtol=0, atol=1e-6)))
    grid = np.linspace(a, b, 4001)
    interpolated = polyn_val(coefficients, grid)
    error = np.max(np.abs(interpolated - f(grid)))
    print(f"{title}: maximum sampled absolute error = {error:.6g}")
    ax.plot(grid, f(grid), color="#146c94", linewidth=2, label="f(t)")
    ax.plot(grid, interpolated, "--", color="#c55232", linewidth=1.8, label="p(t)")
    ax.plot(nodes, values, "o", color="#333333", markersize=3, label="Nodes")
    ax.set(title=title, xlabel="t", ylabel="Value")
    ax.grid(alpha=0.2)
    ax.legend()

print("\nP12 checks:")
for name, passed in checks:
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
print(f"{sum(passed for _, passed in checks)}/{len(checks)} checks passed.")
print("Runge's phenomenon: large oscillations between nodes near the interval ends.")
print("The endpoint nodes themselves are interpolated; the trouble is nearby.")
output = Path(__file__).with_name("p12_interpolation.png")
fig.savefig(output, dpi=180)
print(f"Plot saved to {output}")
plt.show()


