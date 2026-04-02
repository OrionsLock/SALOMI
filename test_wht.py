import numpy as np
from onebit.ops.hadamard import fast_walsh_hadamard

np.random.seed(42)
d = 64

# Test 1: WHT is self-inverse
x = np.random.randn(d).astype(np.float32)
x_h = fast_walsh_hadamard(x, normalize=True)
x_back = fast_walsh_hadamard(x_h, normalize=True)
print('Test 1 - WHT self-inverse:')
print(f'  Max error: {np.max(np.abs(x - x_back)):.2e}')
print(f'  PASS' if np.max(np.abs(x - x_back)) < 1e-5 else '  FAIL')

# Test 2: W @ x = W_h @ x_h where W_h = W @ H, x_h = H @ x
W = np.random.randn(32, d).astype(np.float32)
y_true = W @ x

# Apply H to each row of W
W_h = np.zeros_like(W)
for i in range(W.shape[0]):
    W_h[i] = fast_walsh_hadamard(W[i], normalize=True)

x_h = fast_walsh_hadamard(x, normalize=True)
y_h = W_h @ x_h

print('\nTest 2 - Matmul equivalence (W @ x = W_h @ x_h):')
print(f'  y_true[:3]: {y_true[:3]}')
print(f'  y_h[:3]:    {y_h[:3]}')
print(f'  Max error: {np.max(np.abs(y_true - y_h)):.2e}')
print(f'  PASS' if np.max(np.abs(y_true - y_h)) < 1e-5 else '  FAIL')

# Test 3: Quantization in Hadamard domain
# The idea: sign(W_h) @ x_h should be closer to y_true than sign(W) @ x

W_sign = np.sign(W)
W_sign[W_sign == 0] = 1
y_naive = W_sign @ x

W_h_sign = np.sign(W_h)
W_h_sign[W_h_sign == 0] = 1
y_hadamard = W_h_sign @ x_h

# Correlation with true output
corr_naive = np.corrcoef(y_true, y_naive)[0, 1]
corr_hadamard = np.corrcoef(y_true, y_hadamard)[0, 1]

print('\nTest 3 - Binary quantization quality:')
print(f'  Naive 1-bit correlation:    {corr_naive:.4f}')
print(f'  Hadamard 1-bit correlation: {corr_hadamard:.4f}')
print(f'  Hadamard is {"BETTER" if corr_hadamard > corr_naive else "WORSE"}')

# Test 4: Check magnitude uniformity
print('\nTest 4 - Magnitude distribution:')
print(f'  W magnitudes:   mean={np.mean(np.abs(W)):.4f}, std={np.std(np.abs(W)):.4f}')
print(f'  W_h magnitudes: mean={np.mean(np.abs(W_h)):.4f}, std={np.std(np.abs(W_h)):.4f}')

