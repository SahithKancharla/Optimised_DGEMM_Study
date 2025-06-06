#include "kernel.h"
#include <immintrin.h>

/*
 * 24×6 micro‑kernel – AVX‑512 (double)
 * -----------------------------------------------------------
 *   Fast path  : mr == 24  →  aligned loads/stores, no masks
 *   Slow path  : mr <  24  →  masked loads/stores
 *   Branch‑free kc loop chosen with one switch(nr)
 */

#define MR 24
#define NR  6

static inline __mmask8 row_mask(int rows, int offset)
{
    int r = rows - offset;
    return (r <= 0) ? 0 : (r >= 8 ? (__mmask8)0xFF : (__mmask8)((1u << r) - 1u));
}

#define FMA_P(col, CV, Pb)                                       \
  do {                                                            \
    __m512d bb = _mm512_set1_pd((Pb)[col]);                       \
    (CV)[0] = _mm512_fmadd_pd(a0, bb, (CV)[0]);                   \
    (CV)[1] = _mm512_fmadd_pd(a1, bb, (CV)[1]);                   \
    (CV)[2] = _mm512_fmadd_pd(a2, bb, (CV)[2]);                   \
  } while (0)

#define EMIT_FMAS(n, C0,C1,C2,C3,C4,C5, Pb)                       \
  do {                                                            \
    if ((n) > 0) FMA_P(0, C0, Pb);                                 \
    if ((n) > 1) FMA_P(1, C1, Pb);                                 \
    if ((n) > 2) FMA_P(2, C2, Pb);                                 \
    if ((n) > 3) FMA_P(3, C3, Pb);                                 \
    if ((n) > 4) FMA_P(4, C4, Pb);                                 \
    if ((n) > 5) FMA_P(5, C5, Pb);                                 \
  } while (0)

#define UNROLL_KC_STEP(M)                                         \
  do {                                                            \
    __m512d a0 = _mm512_load_pd(A + (M)*MR +  0);                 \
    __m512d a1 = _mm512_load_pd(A + (M)*MR +  8);                 \
    __m512d a2 = _mm512_load_pd(A + (M)*MR + 16);                 \
    const double *Pb = B + (M)*NR;                                \
    EMIT_FMAS(nr, C0,C1,C2,C3,C4,C5, Pb);                         \
  } while (0)
  
void kernel_24x6(const double *restrict A,
                 const double *restrict B,
                 double *restrict       C,
                 int                    mr,
                 int                    nr,
                 int                    kc,
                 int                    ldC)
{
    /* ========================================================= */
    /* Fast‑path : full 24 rows – no masks                       */
    /* ========================================================= */
    if (mr == MR) {
        __m512d C0[3], C1[3], C2[3], C3[3], C4[3], C5[3];
#define LD_FULL(col, dst)                                                       \
        dst[0] = _mm512_load_pd(C + (size_t)(col) * ldC +  0);                \
        dst[1] = _mm512_load_pd(C + (size_t)(col) * ldC +  8);                \
        dst[2] = _mm512_load_pd(C + (size_t)(col) * ldC + 16);
        switch (nr) {
        default: LD_FULL(5, C5);
        case 5:  LD_FULL(4, C4);
        case 4:  LD_FULL(3, C3);
        case 3:  LD_FULL(2, C2);
        case 2:  LD_FULL(1, C1);
        case 1:  LD_FULL(0, C0);
        }
#undef LD_FULL

#define FMA_COL(col, CV)                                      \
        { __m512d b = _mm512_set1_pd(B[col]);                 \
          CV[0] = _mm512_fmadd_pd(a0, b, CV[0]);              \
          CV[1] = _mm512_fmadd_pd(a1, b, CV[1]);              \
          CV[2] = _mm512_fmadd_pd(a2, b, CV[2]); }

        switch (nr) {
            default: {  /* nr==6 */
                int p;
                for (p = 0; p + 1 < kc; p += 2) {
                    _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                    UNROLL_KC_STEP(0);
                    UNROLL_KC_STEP(1);
                    A += 2*MR;
                    B += 2*NR;
                }
                if (p < kc) {
                    UNROLL_KC_STEP(0);
                    A += MR;
                    B += NR;
                }
            } break;
            case 5:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 4:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 3:
                {
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 2:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 1:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
        }
#undef FMA_COL

#define ST_FULL(col, src)                                             \
        _mm512_store_pd(C + (size_t)(col) * ldC +  0, src[0]);      \
        _mm512_store_pd(C + (size_t)(col) * ldC +  8, src[1]);      \
        _mm512_store_pd(C + (size_t)(col) * ldC + 16, src[2]);
        switch (nr) {
        default: ST_FULL(5, C5);
        case 5:  ST_FULL(4, C4);
        case 4:  ST_FULL(3, C3);
        case 3:  ST_FULL(2, C2);
        case 2:  ST_FULL(1, C1);
        case 1:  ST_FULL(0, C0);
        }
#undef ST_FULL
        return;
    }

    /* ========================================================= */
    /* Slow‑path : mr < 24 – masked version                      */
    /* ========================================================= */

    const __mmask8 m0 = row_mask(mr, 0);
    const __mmask8 m1 = row_mask(mr, 8);
    const __mmask8 m2 = row_mask(mr, 16);

#define LD_C(col, v0,v1,v2)                                      \
    v0 = _mm512_maskz_loadu_pd(m0, C + (size_t)(col) * ldC);     \
    v1 = _mm512_maskz_loadu_pd(m1, C + (size_t)(col) * ldC + 8); \
    v2 = _mm512_maskz_loadu_pd(m2, C + (size_t)(col) * ldC + 16);

#define ST_C(col, v0,v1,v2)                                      \
    _mm512_mask_storeu_pd(C + (size_t)(col) * ldC,      m0, v0); \
    if (m1) _mm512_mask_storeu_pd(C + (size_t)(col) * ldC + 8,  m1, v1); \
    if (m2) _mm512_mask_storeu_pd(C + (size_t)(col) * ldC + 16, m2, v2);

    __m512d C0[3], C1[3], C2[3], C3[3], C4[3], C5[3];
    switch (nr) {
    default: LD_C(5, C5[0], C5[1], C5[2]);
    case 5:  LD_C(4, C4[0], C4[1], C4[2]);
    case 4:  LD_C(3, C3[0], C3[1], C3[2]);
    case 3:  LD_C(2, C2[0], C2[1], C2[2]);
    case 2:  LD_C(1, C1[0], C1[1], C1[2]);
    case 1:  LD_C(0, C0[0], C0[1], C0[2]);
    }
#undef LD_C

#define FMA_MASK(col, CV)                                   \
    {   __m512d b = _mm512_broadcastsd_pd(_mm_load_sd(B + col));                 \
        CV[0] = _mm512_fmadd_pd(a0, b, CV[0]);              \
        CV[1] = _mm512_fmadd_pd(a1, b, CV[1]);              \
        CV[2] = _mm512_fmadd_pd(a2, b, CV[2]); }

        switch (nr) {
            default: {  /* nr==6 */
                int p;
                for (p = 0; p + 1 < kc; p += 2) {
                    _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                    _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                    UNROLL_KC_STEP(0);
                    UNROLL_KC_STEP(1);
                    A += 2*MR;
                    B += 2*NR;
                }
                if (p < kc) {
                    UNROLL_KC_STEP(0);
                    A += MR;
                    B += NR;
                }
            } break;
            case 5:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 4:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 3:
                {
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 2:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
            case 1:
                { 
                    int p;
                    for (p = 0; p + 1 < kc; p += 2) {
                        _mm_prefetch((const char*)(A + 2*MR), _MM_HINT_T0);
                        _mm_prefetch((const char*)(B + 2*NR), _MM_HINT_T0);
                        UNROLL_KC_STEP(0);
                        UNROLL_KC_STEP(1);
                        A += 2*MR;
                        B += 2*NR;
                    }
                    if (p < kc) {
                        UNROLL_KC_STEP(0);
                        A += MR;
                        B += NR;
                    }
                } break;
        }
#undef FMA_MASK

    switch (nr) {
    default: ST_C(5, C5[0], C5[1], C5[2]);
    case 5:  ST_C(4, C4[0], C4[1], C4[2]);
    case 4:  ST_C(3, C3[0], C3[1], C3[2]);
    case 3:  ST_C(2, C2[0], C2[1], C2[2]);
    case 2:  ST_C(1, C1[0], C1[1], C1[2]);
    case 1:  ST_C(0, C0[0], C0[1], C0[2]);
    }
#undef ST_C
}
