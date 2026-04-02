// BSDM-W naive normalized kernel with Walsh N=2 + antithetic, subgroup early-exit, optional CTG
// Normalized domain: u in [-1,1]; outputs Y_mean are sample means of y_bar in {+1,-1}
// Tick schedule: N=2 carriers x 2 antithetic = 4 samples/tick, y_bar = mean(4 samples)

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

inline int popcnt_uint(uint x) { return popcount(x); }

// Parity of bits helper
inline int parity_u(uint x) { return popcount(x) & 1; }

// Walsh carrier bit for row n and tick t: c(n,t) = (-1)^parity(n & t)
inline float walsh_carrier(uint n, uint t) {
    return (parity_u(n & t) == 0) ? 1.0f : -1.0f;
}

// PCG32 - minimal (for CTG), not used when use_ctg==0
typedef struct { ulong state; ulong inc; } pcg32_rng_t;
inline void pcg32_seed(pcg32_rng_t *r, ulong seed, ulong seq) {
    r->state = 0UL; r->inc = (seq << 1U) | 1U;
    // advance once with seed
    ulong old = r->state;
    r->state = old * 6364136223846793005UL + r->inc;
    r->state += seed;
}
inline uint pcg32_rand(pcg32_rng_t *r) {
    ulong old = r->state;
    r->state = old * 6364136223846793005UL + r->inc;
    uint xorshifted = (uint)(((old >> 18U) ^ old) >> 27U);
    uint rot = (uint)(old >> 59U);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// SplitMix64 step (for state initialization)
inline float uniform_half(ulong *state) {
    *state += 0x9E3779B97F4A7C15UL;
    ulong z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z = z ^ (z >> 31);
    float u01 = (float)(z >> 32) / 4294967296.0f;
    return u01 - 0.5f;
}

// SplitMix32 for CTG ops (deterministic, branchless)
inline uint splitmix32(ulong *state) {
    *state += 0x9E3779B97F4A7C15UL;
    ulong z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z = z ^ (z >> 31);
    return (uint)(z >> 32);
}

__kernel void bsdm_w_naive_norm(
  __global const uint *W_bits,      // [M, Kw]
  __global const uint *X_bits,      // [T, Kw] (prepacked per tick)
  const int Kw,                     // words per row
  const int T,                      // max ticks
  const float eps,                  // Hoeffding epsilon
  const float delta,                // per-row delta
  const uchar order,                // 1 or 2
  const float beta,                 // SD-2 beta (used by MASH/leak update)
  const float lambda,               // leak
  const uchar use_ctg,              // 0/1
  const ulong prf_seed,             // PRF seed (CTG; can also tweak init)
  const uchar walsh_N,              // carriers per tick (2 for PR-1.3)
  const uchar antithetic,           // 1 for antithetic pairs
  const uchar early_exit_enable,    // 0=force full T (Stage-A), 1=allow early-exit
  __global float *Y_mean,           // [M]
  __global int   *T_eff,            // [M]
  __global uint  *y_main_pack,      // [M, ceil(T*N/32)] main channel bits
  __global uint  *y_twin_pack,      // [M, ceil(T*N/32)] twin channel bits (if antithetic)
  __global int   *pc32_main,        // [M, T] optional popcount main
  __global int   *pc32_twin,        // [M, T] optional popcount twin
  __global uint  *ctg_digest_out,   // [M] CTG digest (rolling hash of ops)
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
) {
  const int row = get_global_id(0);
  const int M = get_global_size(0);
  if (row >= M) return;

  const __global uint *w = W_bits + (size_t)row * (size_t)Kw;
  const int N = (int)walsh_N;
  const int use_anti = (int)antithetic;

  // SD states per carrier: main and twin channels
  // For N=2: e_main[0], e_main[1], e_twin[0], e_twin[1]
  float e_main[2] = {0.0f, 0.0f};
  float e_twin[2] = {0.0f, 0.0f};
  float e1_main[2] = {0.0f, 0.0f};
  float e2_main[2] = {0.0f, 0.0f};
  float e1_twin[2] = {0.0f, 0.0f};
  float e2_twin[2] = {0.0f, 0.0f};

  // Initialize states with dithering from PRF seed (matches CPU)
  // Derive per-row seed: (prf_seed + row) to match CPU stageA_probe_topT
  ulong init_state = (prf_seed + (ulong)row) & 0xFFFFFFFFFFFFFFFFUL;
  if (order == (uchar)1) {
    for (int n = 0; n < N; n++) {
      e_main[n] = uniform_half(&init_state);
    }
    if (use_anti) {
      for (int n = 0; n < N; n++) {
        e_twin[n] = uniform_half(&init_state);
      }
    }
  } else {
    for (int n = 0; n < N; n++) {
      e1_main[n] = uniform_half(&init_state);
    }
    for (int n = 0; n < N; n++) {
      e2_main[n] = uniform_half(&init_state);
    }
    if (use_anti) {
      for (int n = 0; n < N; n++) {
        e1_twin[n] = uniform_half(&init_state);
      }
      for (int n = 0; n < N; n++) {
        e2_twin[n] = uniform_half(&init_state);
      }
    }
  }

  const float E1 = 4.0f;
  const float E2 = 8.0f;

  // Running accumulators
  int t_used = 0;
  float acc_ybar = 0.0f;

  // Energy instrumentation: private accumulators (PR-3.7)
  ulong p_tog_m = 0, p_tog_t = 0, p_ones = 0, p_zeros = 0;
  ulong p_xnor = 0, p_pop = 0, p_bW = 0, p_bX = 0, p_bOut = 0;
  uchar prev_y_main[2] = {0, 0};  // Previous y bits per carrier
  uchar prev_y_twin[2] = {0, 0};

  // y bit packing base pointers
  const int samples_per_channel = T * N;
  const int words_per_channel = (samples_per_channel + 31) >> 5;
  __global uint *y_main_row = 0;
  __global uint *y_twin_row = 0;
  __global int *pc_main_row = 0;
  __global int *pc_twin_row = 0;

  if (y_main_pack != 0) {
    y_main_row = y_main_pack + (size_t)row * (size_t)words_per_channel;
  }
  if (y_twin_pack != 0 && use_anti) {
    y_twin_row = y_twin_pack + (size_t)row * (size_t)words_per_channel;
  }
  if (pc32_main != 0) {
    pc_main_row = pc32_main + (size_t)row * (size_t)T;
  }
  if (pc32_twin != 0 && use_anti) {
    pc_twin_row = pc32_twin + (size_t)row * (size_t)T;
  }

  // CTG state (SplitMix64 for determinism)
  ulong ctg_state = (prf_seed + (ulong)row) & 0xFFFFFFFFFFFFFFFFUL;
  uint ctg_digest = 0u;  // Rolling hash of CTG ops

  const float invN_bits = 1.0f / (float)(Kw * 32);

  // Iterate ticks
  for (int t = 0; t < T; ++t) {
    // Popcount across Kw words (computed once per tick)
    const __global uint *x = X_bits + (size_t)t * (size_t)Kw;
    int pc = 0;
    for (int i = 0; i < Kw; ++i) {
      const uint xnorw = ~(w[i] ^ x[i]);
      pc += popcnt_uint(xnorw);
    }
    // Normalize to [-1, 1]
    float base_u = ((float)(pc * 2 - Kw * 32)) * invN_bits;

    // Store pc32 if requested (as popcount, not centered)
    if (pc_main_row != 0) {
      pc_main_row[t] = pc;
    }
    if (pc_twin_row != 0 && use_anti) {
      pc_twin_row[t] = pc;
    }

    // Energy: accumulate popcount stats (PR-3.7)
    p_ones += (ulong)pc;
    p_zeros += (ulong)(Kw * 32 - pc);
    p_xnor += (ulong)Kw;
    p_pop += (ulong)Kw;
    p_bW += (ulong)(4 * Kw);
    p_bX += (ulong)(4 * Kw);

    // CTG (Constant-Time Grammar) - deterministic procedural transforms
    // Applied per-tick, after carrier modulation, before SD updates
    // Ops: PASS(00), INVERT(01), INHIBIT(10), PHASE(11)
    uint ctg_op = 0u;
    if (use_ctg) {
      uint r = splitmix32(&ctg_state);
      ctg_op = r & 3u;
      ctg_digest = (ctg_digest << 2) ^ ctg_op;  // Rolling hash
    }

    // Accumulate samples for this tick
    float tick_sum = 0.0f;
    int tick_count = 0;

    // Iterate carriers (N=2)
    for (int n = 0; n < N; ++n) {
      float c = walsh_carrier((uint)n, (uint)t);

      // Carrier-modulated values (before CTG)
      float u_main = base_u * c;
      float u_twin = use_anti ? (-base_u * c) : (base_u * c);

      // Apply CTG transforms (branchless)
      if (use_ctg) {
        // PHASE (11): swap main<->twin
        uint phase_mask = (ctg_op == 3u) ? 1u : 0u;
        float u_m_tmp = phase_mask ? u_twin : u_main;
        float u_t_tmp = phase_mask ? u_main : u_twin;
        u_main = u_m_tmp;
        u_twin = u_t_tmp;

        // INVERT (01): flip signs
        uint invert_mask = (ctg_op == 1u) ? 1u : 0u;
        float sgn = invert_mask ? -1.0f : 1.0f;
        u_main *= sgn;
        u_twin *= sgn;

        // INHIBIT (10): create +v/-v pairing to cancel within tick
        // v = 0.5*(u_main + u_twin) is the carrier-combined sample
        // Set u_main = +v, u_twin = -v so tick average ~= 0
        uint inhibit_mask = (ctg_op == 2u) ? 1u : 0u;
        float v = 0.5f * (u_main + u_twin);
        float keep = (float)(1u - inhibit_mask);
        float inh = (float)inhibit_mask;
        u_main = keep * u_main + inh * v;
        u_twin = keep * u_twin + inh * (-v);
      }

      // Use u_main for main channel SD
      float u = u_main;

      // Main channel
      float y_main;
      int yb_main;
      if (order == (uchar)1) {
        float v = u + e_main[n];
        y_main = (v >= 0.0f) ? 1.0f : -1.0f;
        e_main[n] = e_main[n] + u - y_main - lambda * e_main[n];
        if (e_main[n] > E1) e_main[n] = E1;
        else if (e_main[n] < -E1) e_main[n] = -E1;
        yb_main = (y_main > 0.0f) ? 1 : 0;
      } else {
        float v = u + e1_main[n] + e2_main[n];
        y_main = (v >= 0.0f) ? 1.0f : -1.0f;
        float e1_prev = e1_main[n];
        e1_main[n] = e1_main[n] + u - y_main - lambda * e1_main[n];
        if (e1_main[n] > E1) e1_main[n] = E1;
        else if (e1_main[n] < -E1) e1_main[n] = -E1;
        e2_main[n] = e2_main[n] + e1_main[n] - lambda * e2_main[n] + beta * (e1_main[n] - e1_prev);
        if (e2_main[n] > E2) e2_main[n] = E2;
        else if (e2_main[n] < -E2) e2_main[n] = -E2;
        yb_main = (y_main > 0.0f) ? 1 : 0;
      }

      // Pack main bit
      if (y_main_row != 0 && yb_main) {
        const int bit_idx = t * N + n;
        const int word_idx = bit_idx >> 5;
        const int bit = bit_idx & 31;
        y_main_row[word_idx] |= (1u << bit);
      }

      // Energy: track toggles (PR-3.7)
      p_tog_m += (ulong)((uchar)yb_main ^ prev_y_main[n]);
      prev_y_main[n] = (uchar)yb_main;

      tick_sum += y_main;
      tick_count += 1;

      // Twin channel (antithetic) - use CTG-transformed u_twin
      if (use_anti) {
        // u_twin already computed above with CTG applied
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

        // Pack twin bit
        if (y_twin_row != 0 && yb_twin) {
          const int bit_idx = t * N + n;
          const int word_idx = bit_idx >> 5;
          const int bit = bit_idx & 31;
          y_twin_row[word_idx] |= (1u << bit);
        }

        // Energy: track toggles (PR-3.7)
        p_tog_t += (ulong)((uchar)yb_twin ^ prev_y_twin[n]);
        prev_y_twin[n] = (uchar)yb_twin;

        tick_sum += y_twin;
        tick_count += 1;
      }
    }

    // Compute y_bar for this tick
    float y_bar_t = tick_sum / (float)tick_count;
    acc_ybar += y_bar_t;
    t_used = t + 1;

    // Early-exit bound (mean of y_bar vs 0) - only if enabled
    if (early_exit_enable) {
      float mean_ybar = acc_ybar / (float)t_used;
      float thr = sqrt(0.5f * log(2.0f / fmax(delta, 1e-20f)) / (float)t_used);
      int done_lane = (fabs(mean_ybar) <= (eps + thr)) ? 1 : 0;
#if __OPENCL_C_VERSION__ >= 200
      if (sub_group_all(done_lane)) {
        break;
      }
#else
      if (done_lane) {
        break; // per-thread early exit fallback
      }
#endif
    }
  }

  // Write outputs
  Y_mean[row] = (t_used > 0) ? (acc_ybar / (float)t_used) : 0.0f;
  T_eff[row] = t_used;
  if (ctg_digest_out != 0) {
    ctg_digest_out[row] = ctg_digest;
  }

  // Energy: compute bytes_out and store counters (PR-3.7)
  if (instr_on) {
    ulong bytes_bits = (ulong)((t_used * N + 7) / 8);
    ulong bytes_pc32 = (pc_main_row != 0) ? (ulong)(4 * t_used) : 0UL;
    p_bOut = bytes_bits + bytes_pc32;

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

