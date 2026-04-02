import numpy as np
from onebit.core.packbits import pack_float_to_stream

def check_packer():
    x = np.array([0.5, -0.5, 0.1, -0.9, 0.0])
    T = 1000
    stream = pack_float_to_stream(x, k=T)
    
    # stream is [T, Kw] uint32
    # decode stream
    # Iterate t, extract bits, sum
    decoded = np.zeros_like(x)
    
    for t in range(T):
        # simple unpacking for debug
        row = stream[t]
        bits = []
        for w in row:
            for b in range(32):
                val = 1.0 if (w & (1<<b)) else -1.0
                bits.append(val)
        bits = np.array(bits[:len(x)])
        decoded += bits
        
    decoded /= T
    
    print("Original:", x)
    print("Decoded: ", decoded)
    print("Error:   ", np.abs(x - decoded))

if __name__ == "__main__":
    check_packer()

