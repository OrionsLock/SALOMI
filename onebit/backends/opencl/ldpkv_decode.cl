// LDP-KV decode kernel (Stage-1: compute energies for all positions)
// One work-item per position

// SplitMix64 PRF
inline ulong splitmix64(ulong state) {
    ulong z = state + 0x9e3779b97f4a7c15UL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

__kernel void ldpkv_decode_stage1(
    __global const float* K_enc,        // [d_dim] encoded keys
    __global const int* row_ptr,        // [n_pos + 1] CSR row pointers
    __global const int* col_idx,        // [n_edges] column indices
    __global const char* edge_weights,  // [n_edges] edge weights (±1)
    int n_pos,                          // Number of positions
    int T,                              // Number of ticks
    uchar order,                        // SD modulator order (1 or 2)
    float beta,                         // SD-2 beta parameter
    float lambda,                       // SD leak parameter
    uchar early_exit_enable,            // Enable early-exit
    __global float* E_mean,             // [n_pos] output: mean energies
    __global int* T_eff                 // [n_pos] output: effective ticks used
) {
    int pos = get_global_id(0);
    if (pos >= n_pos) return;
    
    // SD states
    float E1 = 0.0f;
    float E2 = 0.0f;
    float y_sum = 0.0f;
    
    int k_used = 0;
    
    // Get edge range for this position
    int edge_start = row_ptr[pos];
    int edge_end = row_ptr[pos + 1];
    int n_edges = edge_end - edge_start;
    
    for (int t = 0; t < T; t++) {
        // Compute energy via expander graph
        float u = 0.0f;
        
        for (int e = edge_start; e < edge_end; e++) {
            int dim_id = col_idx[e];
            char weight = edge_weights[e];
            
            u += (float)weight * K_enc[dim_id];
        }
        
        // Normalize
        u = u / (float)n_edges;
        
        // SD modulation
        float y;
        if (order == 1) {
            y = (u + E1) >= 0.0f ? 1.0f : -1.0f;
            E1 += lambda * (u - y);
        } else {
            // Order-2 (MASH-1-1 with leak)
            float e1_next = E1 + u;
            float e1_clamped = clamp(e1_next, -4.0f, 4.0f);
            
            float e2_next = E2 + e1_clamped;
            float e2_clamped = clamp(e2_next, -8.0f, 8.0f);
            
            y = e2_clamped >= 0.0f ? 1.0f : -1.0f;
            
            E1 = e1_clamped + lambda * (u - y);
            E2 = e2_clamped - y;
        }
        
        y_sum += y;
        k_used++;
        
        // Early-exit (simplified)
        if (early_exit_enable && t >= 8) {
            break;
        }
    }
    
    // Write outputs
    E_mean[pos] = y_sum / (float)k_used;
    T_eff[pos] = k_used;
}

__kernel void ldpkv_decode_stage2(
    __global const float* V_enc,        // [d_dim] encoded values
    __global const int* row_ptr,        // [n_pos + 1] CSR row pointers
    __global const int* col_idx,        // [n_edges] column indices
    __global const char* edge_weights,  // [n_edges] edge weights (±1)
    __global const int* winner_positions, // [n_winners] positions to decode
    int n_winners,                      // Number of winners
    int d_kv,                           // Dimension of K/V
    int d_kv_words,                     // Number of 32-bit words for d_kv
    __global uint* V_decoded,           // [n_winners, d_kv_words] output: decoded values
    // Repair mode parameters (PR-4.0)
    uchar repair_pass,                  // If 1, perform repair
    int group_idx,                      // Group index to repair
    int group_size,                     // Positions per group
    __global const float* K_enc,        // [d_dim] encoded keys (for repair)
    __global uint* K_bits_inout         // [n_pos, d_kv_words] KV cache to repair in-place
) {
    int winner_idx = get_global_id(0);

    // Repair mode: repair one group
    if (repair_pass) {
        // Only first work-item performs repair
        if (winner_idx != 0) return;

        // Compute group boundaries
        int group_start = group_idx * group_size;
        int group_end = group_start + group_size;

        // Repair each position in the group
        for (int pos = group_start; pos < group_end; pos++) {
            // Get edge range for this position
            int edge_start = row_ptr[pos];
            int edge_end = row_ptr[pos + 1];
            int n_edges = edge_end - edge_start;

            // Compute energy via expander graph
            float k_sum = 0.0f;
            for (int e = edge_start; e < edge_end; e++) {
                int dim_id = col_idx[e];
                char weight = edge_weights[e];
                k_sum += (float)weight * K_enc[dim_id];
            }

            // Normalize
            float k_mean = k_sum / (float)n_edges;

            // Overwrite K bits in-place (deterministic repair)
            for (int j = 0; j < d_kv; j++) {
                int word_idx = j / 32;
                int bit_idx = j % 32;

                // Clear old bit
                int offset = pos * d_kv_words + word_idx;
                atomic_and(&K_bits_inout[offset], ~(1U << bit_idx));

                // Set new bit based on sign
                if (k_mean >= 0.0f) {
                    atomic_or(&K_bits_inout[offset], (1U << bit_idx));
                }
            }
        }

        return;
    }

    // Standard Stage-2: Decode values for winner positions
    if (winner_idx >= n_winners) return;

    int pos = winner_positions[winner_idx];

    // Get edge range for this position
    int edge_start = row_ptr[pos];
    int edge_end = row_ptr[pos + 1];
    int n_edges = edge_end - edge_start;

    // Decode value via expander graph
    float v_sum = 0.0f;
    for (int e = edge_start; e < edge_end; e++) {
        int dim_id = col_idx[e];
        char weight = edge_weights[e];

        v_sum += (float)weight * V_enc[dim_id];
    }

    // Normalize
    float v_mean = v_sum / (float)n_edges;

    // Simple thresholding: sign determines bit
    for (int j = 0; j < d_kv; j++) {
        int word_idx = j / 32;
        int bit_idx = j % 32;

        int offset = winner_idx * d_kv_words + word_idx;

        // Set bit if v_mean >= 0
        if (v_mean >= 0.0f) {
            atomic_or(&V_decoded[offset], (1U << bit_idx));
        }
    }
}

