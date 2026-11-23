import numpy as np
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

def rank(A, tol=1e-6):
    _, pivots = rref(A, tol)
    return len(pivots)

def null_space_basis(A, tol=1e-6):
    R, pivots = rref(A, tol)
    basis = []
    for c in range(R.shape[1]):
        if c not in pivots:
            v= np.zeros(R.shape[1])
            v[c] = 1.0
            for r, pivot_col in enumerate(pivots):
                v[pivot_col] = -R[r,c]
            basis.append(v)
    return basis

def col_space_basis(A, tol = 1e-6):
    orig= np.array(A, dtype=float)
    _, pivots = rref(orig, tol)
    return [orig[:, c]. copy() for c in pivots]

def test():
    points=[]
    A = np.array([[1, 2, -1], [3, -1, 4]])
    R, pivots= rref(A)
    points.append(())
        
def run_checks():
    checks = []
    A = np.array([[1, 2, -1], [3, -1, 4]])
    R, pivots = rref(A)
    checks.append(("First example RREF", np.allclose(R, [[1, 0, 1], [0, 1, -1]])))
    checks.append(("First example pivots", pivots == [0, 1]))

    # Example 6.31, page 31 of the lecture notes.
    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R, pivots = rref(F)
    checks.append(("F RREF", np.allclose(R, [[1, 0, -1], [0, 1, 2], [0, 0, 0]])))
    checks.append(("F pivots", pivots == [0, 1]))
    checks.append(("F null-space basis", np.allclose(null_space_basis(F), [[1, -2, 1]])))
    checks.append(("F column-space basis", np.allclose(col_space_basis(F), [F[:, 0], F[:, 1]])))

    for i in range(20):
        A = rng.standard_normal((5, 7))
        basis = null_space_basis(A)
        columns = col_space_basis(A)
        checks.append((f"Random matrix {i + 1}", all([
            all(np.allclose(A @ v, np.zeros(5)) for v in basis),
            rank(A) + len(basis) == 7,
            rank(A) == np.linalg.matrix_rank(A),
            len(columns) == rank(A),
            np.linalg.matrix_rank(np.array(columns).T) == rank(A),
        ])))

    A = rng.standard_normal((6, 4)) @ rng.standard_normal((4, 8))
    basis = null_space_basis(A)
    checks.append(("Constructed matrix rank is 4", rank(A) == 4))
    checks.append(("Constructed matrix nullity is 4", len(basis) == 4))
    checks.append(("Constructed matrix rank agrees with NumPy",
                   rank(A) == np.linalg.matrix_rank(A)))
    checks.append(("Constructed matrix null-space vectors",
                   all(np.allclose(A @ v, np.zeros(6)) for v in basis)))

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"{passed_count}/{len(checks)} checks passed.")
    return passed_count == len(checks)


if __name__ == "__main__":
    run_checks()