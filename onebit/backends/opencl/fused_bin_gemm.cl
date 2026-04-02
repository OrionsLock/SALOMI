// Portable fused binary GEMM with per-row early-exit (subgroup vote not required since 1 work-item per row)
__kernel void fused_bin_gemm(
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
    float acc_lo = 0.0f, acc_hi = 0.0f;
    int T_eff = 0;
    if (accum_mode!=0 && ACC_LO && ACC_HI && TEFF){
        acc_lo = ACC_LO[row];
        acc_hi = ACC_HI[row];
        T_eff = TEFF[row];
    }
    const __global uint* Wrow = W_bits + row*Kw;

    for(int t=0; t<T; ++t){
        const __global uint* Xrow = X_bits + t*Kw;
        uint pc = 0u;
        #pragma unroll 4
        for(int i=0;i<Kw;++i){
            uint xnor = ~(Wrow[i] ^ Xrow[i]);
            pc += popcount(xnor);
        }
        int Kbits = Kw<<5;
        float dot = (float)((int)(pc<<1) - Kbits);
        dot = dot * cv[0] + cv[1];

        if((t & 1)==0) acc_lo += dot; else acc_hi += dot;
        ++T_eff;

        // Early-exit condition (portable): single work-item per row, so subgroup vote reduces to local boolean
        if (eps_margin > 0.0f) {
            float diff = fabs(acc_lo - acc_hi);
            float rem = (float)(T - 1 - t);
            float max_dev = 2.0f * sqrt(rem * 0.69314718f); // log(2)
            int want_exit = diff > (eps_margin + max_dev);
            if (want_exit) {
                break;
            }
        }
    }
    int denom = T_eff > 0 ? T_eff : 1;
    if (Y) Y[row] = (acc_lo + acc_hi) / (float)denom;
    if (TEFF) TEFF[row] = T_eff;
    if (accum_mode!=0 && ACC_LO && ACC_HI){
        ACC_LO[row] = acc_lo;
        ACC_HI[row] = acc_hi;
    }
}
