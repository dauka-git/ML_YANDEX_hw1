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

def rank_one(u,v,w):
    return np.einsum("i,j,k-> ijk", u,v,w)

def check(name, real, expected):
    passed = np.allclose(real, expected)
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
    
print("Indexing and unfoldings")
T = np.arange(24).reshape(2, 3, 4)
flat = T.reshape(-1)
indexing_ok = True
for i in range(2):
    for j in range(3):
        for k in range(4):
            if T[i, j, k] != flat[12 * i + 4 * j + k]:
                indexing_ok = False
check("Flattening formula for all 24 indices", indexing_ok, True)
U = T.reshape(2, 12)
W = T.transpose(1, 0, 2).reshape(3, 8)
print("First unfolding:\n", U)
print("Second unfolding:\n", W)
print("Unfolding ranks:", rank(U), rank(W))
check("First unfolding has rank 2", rank(U), 2)
check("Second unfolding has rank 2", rank(W), 2)
check("Unfolding ranks agree with NumPy",
        [rank(U), rank(W)],
        [np.linalg.matrix_rank(U), np.linalg.matrix_rank(W)])

print("\nTable 1 operations")
A = rng.standard_normal((3, 4))
B = rng.standard_normal((4, 2))
C = rng.standard_normal((4, 4))
x = rng.standard_normal(3)
y = rng.standard_normal(3)
z = rng.standard_normal(4)
check("Matrix product", np.einsum("ij,jk->ik", A, B), A @ B)
check("Trace", np.einsum("ii->", C), np.trace(C))
check("Dot product", np.einsum("i,i->", x, y), x @ y)
check("Outer product", np.einsum("i,j->ij", x, y), np.outer(x, y))
check("Transpose", np.einsum("ij->ji", A), A.T)
check("Bilinear form", np.einsum("i,ij,j->", x, A, z), x @ A @ z)

print("\nBatch operations: 100 matrices of size 4 by 4")
A_batch = rng.standard_normal((100, 4, 4))
B_batch = rng.standard_normal((100, 4, 4))
traces = np.einsum("bii->b", A_batch)
products = np.einsum("bij,bjk->bik", A_batch, B_batch)
check("Table 1 batch products", products, A_batch @ B_batch)
loop_traces = np.zeros(100)
loop_products = np.zeros((100, 4, 4))
for b in range(100):
    loop_traces[b] = np.trace(A_batch[b])
    loop_products[b] = A_batch[b] @ B_batch[b]
check("Batch traces agree with loop", traces, loop_traces)
check("Batch products agree with loop", products, loop_products)

print("\nKronecker product and row-major vectorization")
A = rng.standard_normal((3, 4))
B = rng.standard_normal((2, 5))
X = rng.standard_normal((4, 5))
direct = (A @ X @ B.T).reshape(-1)
via_kron = np.kron(A, B) @ X.reshape(-1)
check("Vectorization identity", direct, via_kron)
# For n by n factors, the Kronecker matrix is n^2 by n^2.
n = 100
print(f"For 100 by 100 factors: {n ** 4:,} Kronecker entries.")
print(f"Float64 storage: {8 * n ** 4 / 1e6:.0f} MB.")
print(f"Kronecker matrix-vector product: {n ** 4:,} multiplications.")
print(f"Direct AXB.T: {2 * n ** 3:,} multiplications, without forming Kronecker.")

print("\nRank-one tensor slices")
u = rng.standard_normal(3)
v = rng.standard_normal(4)
w = rng.standard_normal(5)
T = rank_one(u, v, w)
uv = np.outer(u, v)
for k in range(len(w)):
    check(f"Slice {k} equals w[{k}] times u v^T", T[:, :, k], w[k] * uv)
print("Each slice satisfies T[i,j,k] = u[i] * v[j] * w[k].")