import numpy as np
from time import perf_counter
rng = np.random.default_rng(0)

def lu(A):
    U = np.array(A, dtype=float)
    n= U.shape[0]
    L = np.eye(n)

    for k in range(n):
        for i in range(k+1, n):
            L[i,k] = U[i,k] / U[k,k]
            U[i, k+1:] = U[i, k+1:] - L[i,k] * U[k,k+1:]
            U[i,k] = 0.0
    return L,U 

def solve_lu(L, U,b):
    n=len(b)
    y=np.zeros(n)
    for i in range(n):
        val = float(b[i])
        for j in range(i):
            val-= L[i,j] * y[j]
        y[i] = val / L[i,i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        val = y[i]
        for j in range(i+1, n):
            val -= U[i,j] * x[j]
        x[i] = val / U [i,i]
    return x

def inv_lu(A):
    L, U =lu(A)
    n= L.shape[0]
    res = np.zeros((n,n))
    for j in range(n):
        
        e = np.zeros(n)
        e[j] = 1.0
        res[:, j]= solve_lu(L, U, e)
    return res

def update_inv(inv, row):
    n=len(row)
    left= np.zeros(n)
    right= np.zeros(n)

    for i in range(n):
        left += inv[:, i] * row[i]
        right += inv[i, :] * row[i]
        
    denom=1.0
    for i in range(n):
        denom += row[i]*left[i]
    
    return inv - left[:, None] * right[None, :] / denom

def check(name, cond):
    print(f"{name}: {'PASS' if cond else 'FAIL'}")
    
    
    
    
    

A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
L, U = lu(A)
check("Example L", np.allclose(L, [[1, 0, 0], [2, 1, 0], [4, 3, 1]]))
check("Example U", np.allclose(U, [[2, 1, 1], [0, 1, 1], [0, 0, 2]]))
check("Example reconstruction", np.allclose(L @ U, A))
check("Example solution", np.allclose(solve_lu(L, U, [5, 13, 31]), [1, 2, 1]))

print("\nRandom 6 by 6 systems")
for i in range(20):
    A = rng.standard_normal((6, 6))
    b = rng.standard_normal(6)
    L, U = lu(A)
    check(f"Matrix {i + 1}: L @ U = A", np.allclose(L @ U, A))
    check(f"System {i + 1}: solution agrees with NumPy",
            np.allclose(solve_lu(L, U, b), np.linalg.solve(A, b)))

print("\nTiny-pivot example")
A = np.array([[1e-17, 1], [1, 1]])
b = np.array([1.0, 2.0])
L, U = lu(A)
x = solve_lu(L, U, b)
print("Without row swaps:", x)
print("Residual A @ x - b:", A @ x - b)
print("The multiplier 1e17 causes rounding to discard small terms.")
print("Back substitution then subtracts nearly equal numbers and returns x[0] = 0.")
L, U = lu(A[[1, 0], :])
pivoted_x = solve_lu(L, U, b[[1, 0]])
print("After swapping both A's rows and b's entries:", pivoted_x)
check("Swapped solution", np.allclose(pivoted_x, np.linalg.solve(A, b)))

print("\n100 successive inverse updates")
X = rng.standard_normal((200, 50))
new_rows = rng.standard_normal((100, 50))
gram = X.T @ X
initial_inverse = inv_lu(gram)
check("Initial inverse", np.allclose(initial_inverse, np.linalg.inv(gram)))

# Both timings exclude the common initial inverse and random-data generation.
updated = initial_inverse.copy()
start = perf_counter()
for row in new_rows:
    updated = update_inv(updated, row)
update_seconds = perf_counter() - start

data = np.zeros((300, 50))
data[:200, :] = X
data[200:, :] = new_rows
start = perf_counter()
for count in range(201, 301):
    current = data[:count, :]
    recomputed = inv_lu(current.T @ current)
recompute_seconds = perf_counter() - start

final_gram = data.T @ data
check("Updated inverse agrees with recomputation", np.allclose(updated, recomputed))
check("Updated inverse agrees with NumPy", np.allclose(updated, np.linalg.inv(final_gram)))
check("Final inverse residual", np.allclose(final_gram @ updated, np.eye(50)))
print("Maximum entrywise difference:", np.max(np.abs(updated - recomputed)))
print(f"100 Sherman-Morrison updates: {update_seconds:.6f} seconds")
print(f"100 inverses from scratch: {recompute_seconds:.6f} seconds")
print(f"Measured speedup: {recompute_seconds / update_seconds:.1f} times")
print("These timings compare our Python implementations, not optimized NumPy solvers.")
    
