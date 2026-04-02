"""Fixed Memory Comparison: Binary with 1.58x more params vs Ternary.

KEY INSIGHT (from ChatGPT):
At fixed memory M:
- Ternary: M/1.58 parameters
- Binary: M/1.0 = 1.58x more parameters!

We should compare:
- Ternary: d_in × d_out weights at 1.58 bpp
- Binary: d_in × (1.58 × d_out) weights at 1.0 bpp

Same memory, different architectures!
"""

import numpy as np
from typing import Tuple


class TernaryBaseline:
    """Ternary at standard dimensions."""
    def __init__(self, d_in: int, d_out: int, zero_pct: float = 0.3):
        self.d_in, self.d_out = d_in, d_out
        self.W = None
        self.scale = 1.0
        self.zero_pct = zero_pct
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        thresh = np.percentile(np.abs(W_opt), self.zero_pct * 100)
        self.W = np.sign(W_opt) * (np.abs(W_opt) > thresh)
        mask = self.W != 0
        if mask.any():
            self.scale = np.mean(np.abs(W_opt[mask]))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W * self.scale).T
    
    def memory_bits(self) -> int:
        return int(self.d_in * self.d_out * 1.58)


class BinaryWider:
    """Binary with 1.58x more output neurons (same memory as ternary)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out_effective = d_out  # What we want to output
        self.d_out_wide = int(d_out * 1.58)  # Actual width
        self.W = None
        self.scale = 1.0
        # Projection from wide to effective (tiny FP matrix, negligible bits)
        self.proj = None
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        # Train wide binary layer
        # First, find optimal projection and wide weights jointly
        
        # Initialize: wide binary weights
        W_wide_opt = np.random.randn(self.d_out_wide, self.d_in).astype(np.float32) * 0.1
        self.W = np.sign(W_wide_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        
        # Optimal projection: minimize ||Y - X @ W.T @ proj.T||
        # This is linear regression from wide features to Y
        H_wide = X @ self.W.T  # (n_samples, d_out_wide)
        self.proj = np.linalg.lstsq(H_wide, Y, rcond=None)[0].T  # (d_out_effective, d_out_wide)
        
        # Now optimize W given proj
        # Y ≈ X @ W.T @ proj.T = X @ (proj @ W).T
        # So effective weight = proj @ W
        # Optimize W to make proj @ W close to optimal
        W_target = np.linalg.lstsq(X, Y, rcond=None)[0].T  # (d_out_effective, d_in)
        
        # W should satisfy: proj @ W ≈ W_target
        # This is another least squares: W = proj.pinv @ W_target
        proj_pinv = np.linalg.pinv(self.proj)
        W_wide_target = proj_pinv @ W_target
        
        self.W = np.sign(W_wide_target).astype(np.float32)
        self.W[self.W == 0] = 1.0
        
        # Recompute optimal projection and scale
        H_wide = X @ self.W.T
        self.proj = np.linalg.lstsq(H_wide, Y, rcond=None)[0].T
        
        # Scale
        Y_pred = self.forward(X)
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        H = X @ self.W.T  # (n, d_out_wide)
        return H @ self.proj.T * self.scale  # (n, d_out_effective)
    
    def memory_bits(self) -> int:
        # Binary weights + small projection matrix
        binary_bits = self.d_in * self.d_out_wide * 1
        proj_bits = self.d_out_effective * self.d_out_wide * 32  # FP32 projection
        return binary_bits + proj_bits


class BinaryDeeper:
    """Binary with 2 layers instead of 1 (uses same memory as ternary)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        # With 1.58x bits, we can have 2 layers of ~0.79x size each
        # Or keep d_out same and add a hidden layer
        self.d_hidden = int(d_out * 0.79)  # Approximate
        
        self.W1 = None  # (d_hidden, d_in)
        self.W2 = None  # (d_out, d_hidden)
        self.scale1 = 1.0
        self.scale2 = 1.0
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        # Two-layer training
        # Initialize randomly
        self.W1 = np.sign(np.random.randn(self.d_hidden, self.d_in)).astype(np.float32)
        self.W2 = np.sign(np.random.randn(self.d_out, self.d_hidden)).astype(np.float32)
        
        # Alternating optimization
        for _ in range(5):
            # Fix W1, optimize W2
            H = np.maximum(0, X @ self.W1.T)  # ReLU hidden
            W2_opt = np.linalg.lstsq(H, Y, rcond=None)[0].T
            self.W2 = np.sign(W2_opt).astype(np.float32)
            self.W2[self.W2 == 0] = 1.0
            
            # Fix W2, optimize W1 (harder, use gradient)
            # Skip for now, random init is okay for comparison
        
        # Scales
        H = np.maximum(0, X @ self.W1.T)
        Y_pred = H @ self.W2.T
        self.scale2 = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        H = np.maximum(0, X @ (self.W1 * self.scale1).T)
        return H @ (self.W2 * self.scale2).T
    
    def memory_bits(self) -> int:
        return (self.d_in * self.d_hidden + self.d_hidden * self.d_out) * 1


class BinaryMoE:
    """Binary Mixture of Experts with hash routing (Idea #1)."""
    def __init__(self, d_in: int, d_out: int, n_experts: int = 4):
        self.d_in = d_in
        self.d_out = d_out
        self.n_experts = n_experts
        
        # Each expert is smaller, total params ≈ 1.58x ternary
        # ternary_params = d_in * d_out
        # total_binary_params = 1.58 * ternary_params
        # params_per_expert = total_binary_params / n_experts
        self.expert_d_out = int(d_out * 1.58 / n_experts)
        
        self.experts = []  # List of (W, scale) for each expert
        self.combine_proj = None  # (d_out, n_experts * expert_d_out)
    
    def _hash_route(self, X: np.ndarray) -> np.ndarray:
        """Deterministic hash routing based on input."""
        # Simple hash: based on sign pattern of first 8 dimensions
        n_hash_dims = min(8, X.shape[1])
        signs = (X[:, :n_hash_dims] > 0).astype(np.int32)
        hash_val = np.sum(signs * (2 ** np.arange(n_hash_dims)), axis=1)
        return hash_val % self.n_experts
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        # Route samples to experts
        routes = self._hash_route(X)
        
        # Train each expert on its subset
        self.experts = []
        for e in range(self.n_experts):
            mask = routes == e
            if mask.sum() < 10:
                # Not enough samples, use all
                X_e, Y_e = X, Y
            else:
                X_e, Y_e = X[mask], Y[mask]
            
            W_opt = np.linalg.lstsq(X_e, Y_e, rcond=None)[0].T
            W_e = np.sign(W_opt[:self.expert_d_out]).astype(np.float32)
            W_e[W_e == 0] = 1.0
            
            Y_pred = X_e @ W_e.T
            scale_e = np.sum(Y_pred * Y_e[:, :self.expert_d_out]) / (np.sum(Y_pred ** 2) + 1e-8)
            
            self.experts.append((W_e, scale_e))
        
        # Combine projection
        all_expert_out = []
        for i, (W_e, s_e) in enumerate(self.experts):
            mask = routes == i
            out = np.zeros((len(X), self.expert_d_out))
            out[mask] = X[mask] @ (W_e * s_e).T
            all_expert_out.append(out)
        
        H = np.hstack(all_expert_out)  # (n_samples, n_experts * expert_d_out)
        self.combine_proj = np.linalg.lstsq(H, Y, rcond=None)[0].T
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        routes = self._hash_route(X)
        all_expert_out = []
        
        for i, (W_e, s_e) in enumerate(self.experts):
            mask = routes == i
            out = np.zeros((len(X), self.expert_d_out))
            out[mask] = X[mask] @ (W_e * s_e).T
            all_expert_out.append(out)
        
        H = np.hstack(all_expert_out)
        return H @ self.combine_proj.T
    
    def memory_bits(self) -> int:
        expert_bits = self.n_experts * self.d_in * self.expert_d_out * 1
        proj_bits = self.d_out * (self.n_experts * self.expert_d_out) * 32
        return expert_bits + proj_bits


class BinaryBasisSelection:
    """Binary basis dictionary with CTG-like selection (Idea #8)."""
    def __init__(self, d_in: int, d_out: int, n_bases: int = 4):
        self.d_in = d_in
        self.d_out = d_out
        self.n_bases = n_bases

        # Each basis is full size, but we only use 1-2 at a time
        # Total storage = n_bases * d_in * d_out bits
        # To match ternary memory: n_bases * 1.0 = 1.58 → n_bases ≈ 1.58
        # But we want more bases for diversity, so use smaller bases
        self.basis_d_out = int(d_out * 1.58 / n_bases)

        self.bases = []  # List of binary weight matrices
        self.coeffs = None  # Combination coefficients

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T

        # Create diverse bases using SVD
        U, S, Vt = np.linalg.svd(W_opt, full_matrices=False)

        self.bases = []
        for k in range(self.n_bases):
            # Each basis emphasizes different singular vectors
            start = k * (self.d_out // self.n_bases)
            end = min(start + self.basis_d_out, self.d_out)

            # Create basis from subset of SVD
            indices = list(range(start, end))
            if len(indices) < self.basis_d_out:
                indices = list(range(self.basis_d_out))

            W_k = U[:, :self.basis_d_out] @ np.diag(S[:self.basis_d_out]) @ Vt[:self.basis_d_out, :]
            W_k_bin = np.sign(W_k[:self.basis_d_out]).astype(np.float32)
            W_k_bin[W_k_bin == 0] = 1.0
            self.bases.append(W_k_bin)

        # Learn combination coefficients
        # Y ≈ sum_k coeff_k * X @ bases[k].T
        # Stack all basis outputs
        H = np.hstack([X @ b.T for b in self.bases])  # (n, n_bases * basis_d_out)
        self.coeffs = np.linalg.lstsq(H, Y, rcond=None)[0].T

    def forward(self, X: np.ndarray) -> np.ndarray:
        H = np.hstack([X @ b.T for b in self.bases])
        return H @ self.coeffs.T

    def memory_bits(self) -> int:
        basis_bits = self.n_bases * self.d_in * self.basis_d_out * 1
        coeff_bits = self.d_out * (self.n_bases * self.basis_d_out) * 32
        return basis_bits + coeff_bits


class BinaryStandard:
    """Standard binary at same dimensions (for reference)."""
    def __init__(self, d_in: int, d_out: int):
        self.d_in, self.d_out = d_in, d_out
        self.W = None
        self.scale = 1.0

    def train(self, X: np.ndarray, Y: np.ndarray):
        W_opt = np.linalg.lstsq(X, Y, rcond=None)[0].T
        self.W = np.sign(W_opt).astype(np.float32)
        self.W[self.W == 0] = 1.0
        Y_pred = X @ self.W.T
        self.scale = np.sum(Y_pred * Y) / (np.sum(Y_pred ** 2) + 1e-8)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ (self.W * self.scale).T

    def memory_bits(self) -> int:
        return self.d_in * self.d_out * 1


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_fixed_memory_experiments():
    print("=" * 80)
    print("FIXED MEMORY COMPARISON: Binary (1.58x params) vs Ternary")
    print("=" * 80)
    print("\nKey insight: At same MEMORY budget, binary can have 58% more parameters!")

    for d in [64, 128]:
        print(f"\n{'='*70}")
        print(f"Dimension: {d}x{d}")
        ternary_mem = int(d * d * 1.58)
        print(f"Memory budget (ternary baseline): {ternary_mem:,} bits = {ternary_mem/8/1024:.2f} KB")
        print(f"{'='*70}")

        np.random.seed(42)
        W_true = np.random.randn(d, d).astype(np.float32) * 0.5
        X_train = np.random.randn(5000, d).astype(np.float32)
        Y_train = X_train @ W_true.T + np.random.randn(5000, d) * 0.1
        X_test = np.random.randn(1000, d).astype(np.float32)
        Y_test = X_test @ W_true.T

        results = {}

        # Ternary baseline
        tern = TernaryBaseline(d, d)
        tern.train(X_train, Y_train)
        Y_pred = tern.forward(X_test)
        results['ternary'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bits': tern.memory_bits()
        }

        # Binary standard (UNFAIR - less memory)
        binst = BinaryStandard(d, d)
        binst.train(X_train, Y_train)
        Y_pred = binst.forward(X_test)
        results['binary_std'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bits': binst.memory_bits()
        }

        # Binary wider
        binw = BinaryWider(d, d)
        binw.train(X_train, Y_train)
        Y_pred = binw.forward(X_test)
        results['binary_wider'] = {
            'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
            'bits': binw.memory_bits()
        }

        # Binary MoE
        for n_exp in [2, 4, 8]:
            moe = BinaryMoE(d, d, n_experts=n_exp)
            moe.train(X_train, Y_train)
            Y_pred = moe.forward(X_test)
            results[f'binary_moe_{n_exp}'] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bits': moe.memory_bits()
            }

        # Binary basis selection
        for n_bases in [2, 4, 8]:
            basis = BinaryBasisSelection(d, d, n_bases=n_bases)
            basis.train(X_train, Y_train)
            Y_pred = basis.forward(X_test)
            results[f'binary_basis_{n_bases}'] = {
                'corr': np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1],
                'bits': basis.memory_bits()
            }

        # Print results
        print(f"\n{'Method':<20} {'Corr':>10} {'Memory':>12} {'vs Tern':>10}")
        print("-" * 55)
        ternary_corr = results['ternary']['corr']
        ternary_bits = results['ternary']['bits']

        for name, data in sorted(results.items(), key=lambda x: -x[1]['corr']):
            vs = (data['corr'] / ternary_corr - 1) * 100
            mem_ratio = data['bits'] / ternary_bits
            marker = "✓ BEATS!" if data['corr'] >= ternary_corr else ""
            print(f"{name:<20} {data['corr']:>10.4f} {mem_ratio:>10.2f}x {vs:>+9.1f}% {marker}")


if __name__ == "__main__":
    run_fixed_memory_experiments()

