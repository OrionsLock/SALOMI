import numpy as np

def check_binary_weight_corr():
    d = 768
    n = 1000
    rng = np.random.default_rng(42)
    
    W = rng.normal(0, 1, (n, d))
    x = rng.normal(0, 1, (d,))
    
    y_true = W @ x
    
    W_sign = np.sign(W)
    y_bin = W_sign @ x
    
    corr = np.corrcoef(y_true, y_bin)[0, 1]
    print(f"Theoretical Binary Weight Correlation: {corr:.4f}")

if __name__ == "__main__":
    check_binary_weight_corr()

