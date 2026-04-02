// Tiled fused binary GEMM (portable): cooperatively load X tiles into local memory.
// NOTE: To remain portable with local memory barriers, this kernel assumes eps_margin==0.
// Host should select the naive kernel when eps_margin>0 (per-row early-exit).

#define TILE_KW 128

__kernel void fused_bin_gemm_tiled(
    __global const uint *restrict W_bits,  // [M, Kw]
    __global const uint *restrict X_bits,  // [T, Kw]
    __constant float *restrict cv,         // [2]
    const int Kw, const int T,
    const float delta, const float eps_margin,
    __global float *restrict Y,            // [M] or NULL
    __global int *restrict TEFF,           // [M] or NULL (in/out if ACC provided)
    __global float *restrict ACC_LO,       // [M] or NULL (in/out)
    __global float *restrict ACC_HI,       // [M] or NULL (in/out)
    const int accum_mode                   // 0: fresh, 1: accumulate into ACC_*
){
    const int row = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    float acc_lo = 0.0f, acc_hi = 0.0f;
    int T_eff = 0;
    if (accum_mode!=0 && ACC_LO && ACC_HI && TEFF){
        acc_lo = ACC_LO[row];
        acc_hi = ACC_HI[row];
        T_eff = TEFF[row];
    }
    const __global uint* Wrow = W_bits + row*Kw;

    __local uint LX[TILE_KW];

    for (int t=0; t<T; ++t){
        const __global uint* Xrow = X_bits + t*Kw;
        uint pc = 0u;
        for (int kw0=0; kw0<Kw; kw0+=TILE_KW){
            int chunk = Kw - kw0; if (chunk > TILE_KW) chunk = TILE_KW;
            // Cooperative load of X tile into local memory
            for (int i=lid; i<chunk; i+=lsize){
                LX[i] = Xrow[kw0 + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // Compute this tile's contribution for this row
            #pragma unroll 4
            for (int i=0; i<chunk; ++i){
                uint xnor = ~(Wrow[kw0 + i] ^ LX[i]);
                pc += popcount(xnor);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        int Kbits = Kw<<5;
        float dot = (float)((int)(pc<<1) - Kbits);
        dot = dot * cv[0] + cv[1];
        if ((t & 1)==0) acc_lo += dot; else acc_hi += dot;
        ++T_eff;

        // IMPORTANT: To avoid barrier divergence, do not per-row break here.
        // Host should choose naive kernel when eps_margin > 0.
    }

    int denom = T_eff > 0 ? T_eff : 1;
    if (Y) Y[row] = (acc_lo + acc_hi) / (float)denom;
    if (TEFF) TEFF[row] = T_eff;
    if (accum_mode!=0 && ACC_LO && ACC_HI){
        ACC_LO[row] = acc_lo;
        ACC_HI[row] = acc_hi;
    }
}

