import numpy as np
from onebit.ops.bsdm_w import popcount_numpy

def check_popcount():
    # Test case: All ones
    x = np.array([0xFFFFFFFF], dtype=np.uint32)
    pc = popcount_numpy(x)
    print(f"All ones: {pc[0]} (expected 32)")
    
    # Test case: Alternating
    x = np.array([0xAAAAAAAA], dtype=np.uint32)
    pc = popcount_numpy(x)
    print(f"0xAAAAAAAA: {pc[0]} (expected 16)")
    
    # Test case: Random
    rng = np.random.default_rng(42)
    x = rng.integers(0, 0xFFFFFFFF, size=1000, dtype=np.uint32)
    pc = popcount_numpy(x)
    
    # Verify with python bit_count
    errors = 0
    for i in range(1000):
        val = int(x[i])
        ref = val.bit_count()
        if pc[i] != ref:
            errors += 1
            print(f"Error at {i}: Val={val:x}, Calc={pc[i]}, Ref={ref}")
            
    print(f"Total Errors: {errors}")

if __name__ == "__main__":
    check_popcount()

