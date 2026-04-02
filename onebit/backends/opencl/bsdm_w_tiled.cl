// Tiled BSDM-W kernel for OpenCL
// Matches naive kernel exactly when early_exit_enable=0

#define TILE_KW_WORDS 64  // Tile size for Kw dimension (64 words = 2048 dims)

// CTG operations
#define CTG_PASS    0
#define CTG_INVERT  1
#define CTG_INHIBIT 2
#define CTG_PHASE   3

// SplitMix64 for PRF
inline ulong splitmix64(ulong state) {
    ulong z = state + 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

// Popcount for uint
inline uint popcount_u32(uint x) {
    x = x - ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3FU;
}

// Note: uniform_half is defined in bsdm_w_naive_norm.cl

__kernel void bsdm_w_tiled(
    __global const uint *W_bits,  // [M, Kw] - weight matrix (packed bits)
    __global const uint *X_bits,  // [T, Kw] - input matrix (packed bits per tick)
    int Kw,                       // Number of 32-bit words per row
    int T,                        // Number of ticks
    float eps,                    // Effect size (unused in tiled, no early-exit)
    float delta,                  // Risk budget (unused in tiled, no early-exit)
    uchar order,                  // Sigma-Delta order (1 or 2)
    float beta,                   // SD-2 beta parameter
    float lambda,                 // SD leak parameter
    uchar use_ctg,                // CTG enable flag
    ulong prf_seed,               // PRF seed for CTG
    uchar early_exit_enable,      // Must be 0 for tiled
    __global float *Y_mean,       // [M] - output means
    __global int *T_eff,          // [M] - effective ticks used (always T for tiled)
    __global uint *y_pack_main,   // [M, words_per_channel] - main channel bits
    __global uint *y_pack_twin,   // [M, words_per_channel] - twin channel bits
    __global int *pc32_main,      // [M, T] - popcount per tick (main channel)
    __global int *pc32_twin,      // [M, T] - popcount per tick (twin channel)
    __global uint *ctg_digest_out,// [M] - CTG digest per row
    // Energy instrumentation (PR-3.7)
    const uchar instr_on,
    __global ulong *c_toggles_y_main,
    __global ulong *c_toggles_y_twin,
    __global ulong *c_ones_pc,
    __global ulong *c_zeros_pc,
    __global ulong *c_xnor_ops,
    __global ulong *c_popcnt_ops,
    __global ulong *c_bytes_W,
    __global ulong *c_bytes_X,
    __global ulong *c_bytes_out
)
{
    const int row = get_global_id(0);
    const int M = get_global_size(0);
    
    if (row >= M) return;
    
    // Local memory for X tile
    __local uint LX[TILE_KW_WORDS];
    
    // Weight row pointer
    __global const uint *w = W_bits + row * Kw;
    
    // Walsh N is fixed at 2 for Tier-2/3
    const int walsh_N = 2;
    
    // SD state per carrier (N=2): main and twin channels
    // For order 1: use e_main[], e_twin[]
    // For order 2: use e1_main[], e2_main[], e1_twin[], e2_twin[]
    float e_main[2] = {0.0f, 0.0f};
    float e_twin[2] = {0.0f, 0.0f};
    float e1_main[2] = {0.0f, 0.0f};
    float e2_main[2] = {0.0f, 0.0f};
    float e1_twin[2] = {0.0f, 0.0f};
    float e2_twin[2] = {0.0f, 0.0f};
    const float E1 = 4.0f, E2 = 8.0f;
    
    // Initialize SD states with dithering (matches naive kernel exactly)
    ulong init_state = (prf_seed + (ulong)row) & 0xFFFFFFFFFFFFFFFFUL;
    if (order == (uchar)1) {
        for (int n = 0; n < 2; n++) {
            e_main[n] = uniform_half(&init_state);
        }
        for (int n = 0; n < 2; n++) {
            e_twin[n] = uniform_half(&init_state);
        }
    } else {
        for (int n = 0; n < 2; n++) {
            e1_main[n] = uniform_half(&init_state);
        }
        for (int n = 0; n < 2; n++) {
            e2_main[n] = uniform_half(&init_state);
        }
        for (int n = 0; n < 2; n++) {
            e1_twin[n] = uniform_half(&init_state);
        }
        for (int n = 0; n < 2; n++) {
            e2_twin[n] = uniform_half(&init_state);
        }
    }
    
    // CTG state
    ulong ctg_state = (prf_seed + (ulong)row) & 0xFFFFFFFFFFFFFFFFUL;
    uint ctg_digest = 0;

    // Bit packing state
    const int samples_per_channel = T * walsh_N;
    const int words_per_channel = (samples_per_channel + 31) / 32;

    // Energy instrumentation: private accumulators (PR-3.7)
    ulong p_tog_m = 0, p_tog_t = 0, p_ones = 0, p_zeros = 0;
    ulong p_xnor = 0, p_pop = 0, p_bW = 0, p_bX = 0, p_bOut = 0;
    uchar prev_y_main[2] = {0, 0};  // Previous y bits per carrier
    uchar prev_y_twin[2] = {0, 0};
    
    // Accumulator for averaging
    float acc_ybar = 0.0f;
    int t_used = 0;
    
    // Normalization constant
    const int total_bits = Kw * 32;
    const float inv_total_bits = 1.0f / (float)total_bits;
    
    // Process each tick
    for (int t = 0; t < T; t++) {
        // Tile over Kw dimension and accumulate popcount
        uint pc_total = 0u;

        for (int tile_start = 0; tile_start < Kw; tile_start += TILE_KW_WORDS) {
            int tile_size = min(TILE_KW_WORDS, Kw - tile_start);

            // Load X tile into local memory (coalesced)
            int lid = get_local_id(0);
            int lsize = get_local_size(0);
            for (int i = lid; i < tile_size; i += lsize) {
                LX[i] = X_bits[t * Kw + tile_start + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Compute popcount for this tile
            __global const uint *W_row_tile = w + tile_start;
            for (int i = 0; i < tile_size; i++) {
                uint xnor_val = ~(W_row_tile[i] ^ LX[i]);
                pc_total += popcount_u32(xnor_val);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store raw popcount (for diagnostics/golden matrix)
        if (pc32_main != 0) {
            pc32_main[row * T + t] = (int)pc_total;
        }
        if (pc32_twin != 0) {
            pc32_twin[row * T + t] = (int)pc_total;
        }

        // Energy: accumulate popcount stats (no branch on instr_on)
        p_ones += (ulong)pc_total;
        p_zeros += (ulong)(total_bits - pc_total);
        p_xnor += (ulong)Kw;
        p_pop += (ulong)Kw;
        p_bW += (ulong)(4 * Kw);
        p_bX += (ulong)(4 * Kw);

        // Normalize to [-1, 1]: base_u = (2*pc - total_bits) / total_bits
        float base_u = ((float)((int)(pc_total << 1) - (int)total_bits)) * inv_total_bits;

        // Accumulate samples for this tick (4 samples: 2 carriers x main/twin)
        int tick_sum = 0;

        // Walsh N=2 + antithetic (matches naive kernel exactly)
        for (int n = 0; n < 2; ++n) {
            // Walsh carrier: (-1)^parity(n & t)
            float c = (popcount_u32((uint)n & (uint)t) & 1) ? -1.0f : 1.0f;
            
            // Carrier-modulated values
            float u_main = base_u * c;
            float u_twin = -base_u * c;  // Antithetic

            // CTG transforms (if enabled) - applied BEFORE Sigma-Delta
            if (use_ctg) {
                ctg_state = splitmix64(ctg_state);
                uchar ctg_op = (uchar)(ctg_state & 3);
                ctg_digest ^= (uint)ctg_op << (((t * 2 + n) % 16) * 2);
                
                // Apply CTG transform to u values
                if (ctg_op == 1) {  // INVERT
                    u_main = -u_main;
                    u_twin = -u_twin;
                } else if (ctg_op == 2) {  // INHIBIT
                    u_twin = -u_main;
                } else if (ctg_op == 3) {  // PHASE (swap carriers)
                    float tmp = u_main;
                    u_main = u_twin;
                    u_twin = tmp;
                }
            }
            
            // Sigma-Delta modulation (main channel) - use u_main as input
            float y_main;
            int yb_main;
            if (order == (uchar)1) {
                float v = u_main + e_main[n];
                y_main = (v >= 0.0f) ? 1.0f : -1.0f;
                e_main[n] = e_main[n] + u_main - y_main - lambda * e_main[n];
                if (e_main[n] > E1) e_main[n] = E1;
                else if (e_main[n] < -E1) e_main[n] = -E1;
                yb_main = (y_main > 0.0f) ? 1 : 0;
            } else {
                float v = u_main + e1_main[n] + e2_main[n];
                y_main = (v >= 0.0f) ? 1.0f : -1.0f;
                float e1_prev = e1_main[n];
                e1_main[n] = e1_main[n] + u_main - y_main - lambda * e1_main[n];
                if (e1_main[n] > E1) e1_main[n] = E1;
                else if (e1_main[n] < -E1) e1_main[n] = -E1;
                e2_main[n] = e2_main[n] + e1_main[n] - lambda * e2_main[n] + beta * (e1_main[n] - e1_prev);
                if (e2_main[n] > E2) e2_main[n] = E2;
                else if (e2_main[n] < -E2) e2_main[n] = -E2;
                yb_main = (y_main > 0.0f) ? 1 : 0;
            }
            
            // Pack main bit (LSB-first)
            if (y_pack_main != 0 && yb_main) {
                const int bit_idx = t * 2 + n;
                const int word_idx = bit_idx >> 5;
                const int bit = bit_idx & 31;
                if (word_idx < words_per_channel) {
                    y_pack_main[row * words_per_channel + word_idx] |= (1u << bit);
                }
            }

            // Energy: track toggles (no branch on instr_on)
            p_tog_m += (ulong)((uchar)yb_main ^ prev_y_main[n]);
            prev_y_main[n] = (uchar)yb_main;

            tick_sum += (int)y_main;

            // Sigma-Delta modulation (twin channel) - use u_twin as input
            float y_twin;
            int yb_twin;
            if (order == (uchar)1) {
                float v = u_twin + e_twin[n];
                y_twin = (v >= 0.0f) ? 1.0f : -1.0f;
                e_twin[n] = e_twin[n] + u_twin - y_twin - lambda * e_twin[n];
                if (e_twin[n] > E1) e_twin[n] = E1;
                else if (e_twin[n] < -E1) e_twin[n] = -E1;
                yb_twin = (y_twin > 0.0f) ? 1 : 0;
            } else {
                float v = u_twin + e1_twin[n] + e2_twin[n];
                y_twin = (v >= 0.0f) ? 1.0f : -1.0f;
                float e1_prev = e1_twin[n];
                e1_twin[n] = e1_twin[n] + u_twin - y_twin - lambda * e1_twin[n];
                if (e1_twin[n] > E1) e1_twin[n] = E1;
                else if (e1_twin[n] < -E1) e1_twin[n] = -E1;
                e2_twin[n] = e2_twin[n] + e1_twin[n] - lambda * e2_twin[n] + beta * (e1_twin[n] - e1_prev);
                if (e2_twin[n] > E2) e2_twin[n] = E2;
                else if (e2_twin[n] < -E2) e2_twin[n] = -E2;
                yb_twin = (y_twin > 0.0f) ? 1 : 0;
            }
            
            // Pack twin bit (LSB-first)
            if (y_pack_twin != 0 && yb_twin) {
                const int bit_idx = t * 2 + n;
                const int word_idx = bit_idx >> 5;
                const int bit = bit_idx & 31;
                if (word_idx < words_per_channel) {
                    y_pack_twin[row * words_per_channel + word_idx] |= (1u << bit);
                }
            }

            // Energy: track toggles (no branch on instr_on)
            p_tog_t += (ulong)((uchar)yb_twin ^ prev_y_twin[n]);
            prev_y_twin[n] = (uchar)yb_twin;

            tick_sum += (int)y_twin;
        }

        // Per-tick average (4 samples: 2 carriers x main/twin)
        acc_ybar += (float)tick_sum * 0.25f;
        t_used = t + 1;
    }

    // Average over all ticks
    Y_mean[row] = (t_used > 0) ? (acc_ybar / (float)t_used) : 0.0f;
    T_eff[row] = t_used;

    // Write CTG digest
    if (ctg_digest_out != 0) {
        ctg_digest_out[row] = ctg_digest;
    }

    // Energy: compute bytes_out and store counters (PR-3.7)
    // bytes_out = packed bits + pc32 if requested
    ulong bytes_bits = (ulong)((samples_per_channel + 7) / 8);
    ulong bytes_pc32 = (pc32_main != 0) ? (ulong)(4 * T) : 0UL;
    p_bOut = bytes_bits + bytes_pc32;

    // Store counters if instrumentation is on (no branch in hot loop)
    if (instr_on) {
        c_toggles_y_main[row] = p_tog_m;
        c_toggles_y_twin[row] = p_tog_t;
        c_ones_pc[row] = p_ones;
        c_zeros_pc[row] = p_zeros;
        c_xnor_ops[row] = p_xnor;
        c_popcnt_ops[row] = p_pop;
        c_bytes_W[row] = p_bW;
        c_bytes_X[row] = p_bX;
        c_bytes_out[row] = p_bOut;
    }
}

