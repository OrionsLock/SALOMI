// HCL (Hadamard Code Logits) tiled kernel
// Tile-parallel processing with local memory optimization
// Matches naive kernel exactly when early_exit_enable=0

#define TILE_D_WORDS 64  // Tile size for dimension (64 words = 2048 dims)

// SplitMix64 PRF
inline ulong splitmix64(ulong state) {
    ulong z = state + 0x9e3779b97f4a7c15UL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

inline uint splitmix32(ulong state) {
    return (uint)(splitmix64(state) >> 32);
}

// Gray code
inline uint gray_index(uint i) {
    return i ^ (i >> 1);
}

// Hadamard sign word: compute word_bits dimensions starting at word_idx*32
inline uint hadamard_sign_word(int v_id, int word_idx) {
    uint word = 0;
    int base_dim = word_idx * 32;
    
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int j = base_dim + k;
        uint gray_j = gray_index(j);
        
        // Compute popcount(v_id & gray_j)
        uint masked = (uint)v_id & gray_j;
        uint pc = popcount(masked);
        
        // If popcount is even, sign is +1 -> bit = 1
        if ((pc & 1) == 0) {
            word |= (1U << k);
        }
    }
    
    return word;
}

// CTG operators (branchless)
inline float apply_ctg(float u, uchar ctg_op) {
    // PASS (00): u
    // INVERT (01): -u
    // INHIBIT (10): u (not applicable for scalar)
    // PHASE (11): u (not applicable for scalar)
    
    float result = u;
    if (ctg_op == 1) {
        result = -u;
    }
    return result;
}

__kernel void hcl_tiled(
    __global const uint* Q_bits,        // [d_words] query vector (packed bits)
    __global const int* V_ids,          // [Kc] candidate token IDs
    int Kc,                              // Number of candidates
    int d,                               // Dimension
    int d_words,                         // Number of 32-bit words for dimension d
    int T,                               // Number of ticks
    float eps,                           // Early-exit epsilon (unused in tiled)
    float delta,                         // Early-exit delta (unused in tiled)
    uchar order,                         // SD modulator order (1 or 2)
    float beta,                          // SD-2 beta parameter
    float lambda,                        // SD leak parameter
    uchar use_ctg,                       // Enable CTG (0 or 1)
    ulong prf_seed,                      // PRF seed
    uchar early_exit_enable,             // Must be 0 for tiled
    __global float* E_mean,              // [Kc] output: mean energies
    __global int* T_eff,                 // [Kc] output: effective ticks used
    __global uint* y_pack_main,          // [Kc * k_words] optional: packed bits
    __global uint* ctg_digest_out        // [Kc] output: CTG digest per candidate
) {
    const int v_idx = get_global_id(0);
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    
    if (v_idx >= Kc) return;
    
    int v_id = V_ids[v_idx];
    
    // Local memory for Q_bits tile
    __local uint LQ[TILE_D_WORDS];
    
    // Derive per-candidate seed
    ulong seed = prf_seed ^ (0xBF58476D1CE4E5B9UL ^ ((ulong)v_id << 5));
    ulong rng_state = splitmix64(seed);
    
    // CTG state
    ulong ctg_state = 0;
    uint ctg_digest = 0;
    if (use_ctg) {
        ctg_state = splitmix64(seed ^ (ulong)v_idx);
    }
    
    // SD states
    float E1 = 0.0f;
    float E2 = 0.0f;
    float y_sum = 0.0f;
    
    int k_used = 0;
    
    // Process ticks
    for (int t = 0; t < T; t++) {
        // Compute XNOR-popcount between Q_bits and Hadamard row
        int pc = 0;
        
        // Process in tiles
        for (int tile_start = 0; tile_start < d_words; tile_start += TILE_D_WORDS) {
            int tile_end = min(tile_start + TILE_D_WORDS, d_words);
            int tile_size = tile_end - tile_start;
            
            // Cooperative load of Q_bits tile into local memory
            for (int i = local_id; i < tile_size; i += local_size) {
                LQ[i] = Q_bits[tile_start + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Compute popcount for this tile
            for (int i = 0; i < tile_size; i++) {
                int word_idx = tile_start + i;
                
                // Generate Hadamard code word on-the-fly
                uint code_word = hadamard_sign_word(v_id, word_idx);
                
                // XNOR-popcount
                uint xnor = ~(LQ[i] ^ code_word);
                pc += popcount(xnor);
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Normalize to [-1, 1]
        float u = (2.0f * (float)pc - (float)d) / (float)d;
        
        // Apply CTG
        if (use_ctg) {
            ulong ctg_next = splitmix64(ctg_state);
            uint r = splitmix32(ctg_state);
            ctg_state = ctg_next;
            
            uchar ctg_op = (uchar)(r & 3);
            ctg_digest = (ctg_digest << 2) ^ ctg_op;
            
            u = apply_ctg(u, ctg_op);
        }
        
        // SD modulation
        float y;
        if (order == 1) {
            // Order-1: y = sign(u + E1), E1 += leak*(u - y)
            y = (u + E1) >= 0.0f ? 1.0f : -1.0f;
            E1 += lambda * (u - y);
        } else {
            // Order-2 (MASH-1-1 with leak)
            float e1_next = E1 + u;
            float e1_clamped = clamp(e1_next, -4.0f, 4.0f);
            
            float e2_next = E2 + e1_clamped;
            float e2_clamped = clamp(e2_next, -8.0f, 8.0f);
            
            y = e2_clamped >= 0.0f ? 1.0f : -1.0f;
            
            // Update with leak
            E1 = e1_clamped + lambda * (u - y);
            E2 = e2_clamped - y;
        }
        
        // Accumulate
        y_sum += y;
        
        // Pack bits if buffer provided
        if (y_pack_main != 0) {
            int bit_idx = t;
            int word_idx = bit_idx / 32;
            int bit_pos = bit_idx % 32;
            
            int k_words = (T + 31) / 32;
            int offset = v_idx * k_words + word_idx;
            
            if (y > 0.0f) {
                atomic_or(&y_pack_main[offset], (1U << bit_pos));
            }
        }
        
        k_used++;
    }
    
    // Write outputs
    E_mean[v_idx] = y_sum / (float)k_used;
    T_eff[v_idx] = k_used;
    
    if (ctg_digest_out != 0) {
        ctg_digest_out[v_idx] = ctg_digest;
    }
}

