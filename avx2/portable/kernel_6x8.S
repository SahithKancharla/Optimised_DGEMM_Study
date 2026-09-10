/* Hand-scheduled AVX2 DGEMM microkernel, MR=6 x NR=8 (matches BLIS's real
 * production Haswell shape -- reference/blis/bli_family_haswell.h). This
 * is a genuinely separate method from the C-intrinsics ablation series
 * (matmul_v1..v7 in avx2/): those let the compiler choose register
 * allocation and instruction scheduling; this hand-writes both, plus
 * explicit software pipelining and prefetch, the same way OpenBLAS's real
 * dgemm_kernel_4x8_haswell.S (reference/openblas/) and BLIS's
 * bli_gemm_haswell_asm_d6x8.c do.
 *
 * void kernel_6x8_asm(const double *A, const double *B, double *C,
 *                      long kc, long ldC_bytes);
 *
 * A: packed panel, MR=6 doubles per k-step, contiguous (48 bytes/step).
 * B: packed panel, NR=8 doubles per k-step, contiguous (64 bytes/step).
 * C: MR x NR tile, column-major, ldC_bytes = leading dimension IN BYTES
 *    (caller pre-multiplies by sizeof(double) so this kernel does no
 *    scaling -- fewer instructions in the write-back).
 * kc: number of k-steps (panel depth for this call).
 *
 * SysV AMD64 ABI: rdi=A, rsi=B, rdx=C, rcx=kc, r8=ldC_bytes.
 *
 * Register plan (16 YMM total):
 *   ymm3..ymm14  : 12 accumulators, acc[row][colgroup], row=0..5, colgroup=0..1
 *                  ymm3=acc00 ymm4=acc01 ymm5=acc10 ymm6=acc11
 *                  ymm7=acc20 ymm8=acc21 ymm9=acc30 ymm10=acc31
 *                  ymm11=acc40 ymm12=acc41 ymm13=acc50 ymm14=acc51
 *   ymm0         : A broadcast (scratch, reused every row)
 *   ymm1, ymm2   : B vectors b0 (cols 0-3), b1 (cols 4-7)
 *   ymm15        : transpose scratch in the write-back
 *
 * k-loop unrolled by 2 (mirrors OpenBLAS's software pipelining): one full
 * k-step is 48 bytes of A / 64 bytes of B; the unrolled pair advances the
 * pointers once by 96/128 bytes instead of twice by 48/64, halving loop
 * branch/compare overhead. Prefetch reaches PF_DIST steps ahead.
 */
    .text
    .globl kernel_6x8_asm
    .type kernel_6x8_asm,@function
    .align 16
kernel_6x8_asm:
    push    %rbp
    mov     %rsp, %rbp
    sub     $128, %rsp          /* scratch space for the MR=6 remainder rows */

    vxorpd  %ymm3, %ymm3, %ymm3
    vmovapd %ymm3, %ymm4
    vmovapd %ymm3, %ymm5
    vmovapd %ymm3, %ymm6
    vmovapd %ymm3, %ymm7
    vmovapd %ymm3, %ymm8
    vmovapd %ymm3, %ymm9
    vmovapd %ymm3, %ymm10
    vmovapd %ymm3, %ymm11
    vmovapd %ymm3, %ymm12
    vmovapd %ymm3, %ymm13
    vmovapd %ymm3, %ymm14

    movq    %rcx, %r9           /* r9 = remaining k-steps */

.Lk2_loop:
    cmpq    $2, %r9
    jl      .Lk1_check

    prefetcht0  768(%rdi)
    prefetcht0  768(%rsi)

    /* ---- k-step 0 of the pair ---- */
    vmovapd   0(%rsi), %ymm1
    vmovapd  32(%rsi), %ymm2
    vbroadcastsd  0(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm3
    vfmadd231pd %ymm2, %ymm0, %ymm4
    vbroadcastsd  8(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm5
    vfmadd231pd %ymm2, %ymm0, %ymm6
    vbroadcastsd 16(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm7
    vfmadd231pd %ymm2, %ymm0, %ymm8
    vbroadcastsd 24(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm9
    vfmadd231pd %ymm2, %ymm0, %ymm10
    vbroadcastsd 32(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm11
    vfmadd231pd %ymm2, %ymm0, %ymm12
    vbroadcastsd 40(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm13
    vfmadd231pd %ymm2, %ymm0, %ymm14

    /* ---- k-step 1 of the pair ---- */
    vmovapd  64(%rsi), %ymm1
    vmovapd  96(%rsi), %ymm2
    vbroadcastsd 48(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm3
    vfmadd231pd %ymm2, %ymm0, %ymm4
    vbroadcastsd 56(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm5
    vfmadd231pd %ymm2, %ymm0, %ymm6
    vbroadcastsd 64(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm7
    vfmadd231pd %ymm2, %ymm0, %ymm8
    vbroadcastsd 72(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm9
    vfmadd231pd %ymm2, %ymm0, %ymm10
    vbroadcastsd 80(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm11
    vfmadd231pd %ymm2, %ymm0, %ymm12
    vbroadcastsd 88(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm13
    vfmadd231pd %ymm2, %ymm0, %ymm14

    addq    $96, %rdi
    addq    $128, %rsi
    subq    $2, %r9
    jmp     .Lk2_loop

.Lk1_check:
    cmpq    $1, %r9
    jl      .Lwriteback

    vmovapd   0(%rsi), %ymm1
    vmovapd  32(%rsi), %ymm2
    vbroadcastsd  0(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm3
    vfmadd231pd %ymm2, %ymm0, %ymm4
    vbroadcastsd  8(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm5
    vfmadd231pd %ymm2, %ymm0, %ymm6
    vbroadcastsd 16(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm7
    vfmadd231pd %ymm2, %ymm0, %ymm8
    vbroadcastsd 24(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm9
    vfmadd231pd %ymm2, %ymm0, %ymm10
    vbroadcastsd 32(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm11
    vfmadd231pd %ymm2, %ymm0, %ymm12
    vbroadcastsd 40(%rdi), %ymm0
    vfmadd231pd %ymm1, %ymm0, %ymm13
    vfmadd231pd %ymm2, %ymm0, %ymm14

.Lwriteback:
    /* rdx=C, r8=ldC_bytes. Column-group 0 (C cols 0-3): rows 0-3 via 4x4
     * in-register transpose, rows 4-5 via stack scratch (MR=6 isn't a
     * multiple of 4, same reasoning as the C intrinsics write-back). */
    vunpcklpd %ymm5, %ymm3, %ymm0     /* t0 = lo(acc00,acc10) */
    vunpckhpd %ymm5, %ymm3, %ymm1     /* t1 = hi(acc00,acc10) */
    vunpcklpd %ymm9, %ymm7, %ymm2     /* t2 = lo(acc20,acc30) */
    vunpckhpd %ymm9, %ymm7, %ymm15    /* t3 = hi(acc20,acc30) */
    vperm2f128 $0x20, %ymm2, %ymm0, %ymm3   /* col0 */
    vperm2f128 $0x20, %ymm15, %ymm1, %ymm5  /* col1 */
    vperm2f128 $0x31, %ymm2, %ymm0, %ymm7   /* col2 */
    vperm2f128 $0x31, %ymm15, %ymm1, %ymm9  /* col3 */

    movq    %rdx, %rax
    vaddpd  (%rax), %ymm3, %ymm3
    vmovupd %ymm3, (%rax)
    addq    %r8, %rax
    vaddpd  (%rax), %ymm5, %ymm5
    vmovupd %ymm5, (%rax)
    addq    %r8, %rax
    vaddpd  (%rax), %ymm7, %ymm7
    vmovupd %ymm7, (%rax)
    addq    %r8, %rax
    vaddpd  (%rax), %ymm9, %ymm9
    vmovupd %ymm9, (%rax)

    /* Column-group 1 (C cols 4-7): rows 0-3 from acc01,acc11,acc21,acc31 */
    vunpcklpd %ymm6, %ymm4, %ymm0
    vunpckhpd %ymm6, %ymm4, %ymm1
    vunpcklpd %ymm10, %ymm8, %ymm2
    vunpckhpd %ymm10, %ymm8, %ymm15
    vperm2f128 $0x20, %ymm2, %ymm0, %ymm4   /* col0 (C col 4) */
    vperm2f128 $0x20, %ymm15, %ymm1, %ymm6  /* col1 (C col 5) */
    vperm2f128 $0x31, %ymm2, %ymm0, %ymm8   /* col2 (C col 6) */
    vperm2f128 $0x31, %ymm15, %ymm1, %ymm10 /* col3 (C col 7) */

    movq    %rdx, %rax
    leaq    (%rax,%r8,4), %rax        /* rax = C + 4*ldC_bytes (col 4) */
    vaddpd  (%rax), %ymm4, %ymm4
    vmovupd %ymm4, (%rax)
    addq    %r8, %rax
    vaddpd  (%rax), %ymm6, %ymm6
    vmovupd %ymm6, (%rax)
    addq    %r8, %rax
    vaddpd  (%rax), %ymm8, %ymm8
    vmovupd %ymm8, (%rax)
    addq    %r8, %rax
    vaddpd  (%rax), %ymm10, %ymm10
    vmovupd %ymm10, (%rax)

    /* Rows 4-5, both column-groups: spill to stack, scalar scatter (2 rows
     * only -- negligible next to the KC-deep k-loop above). */
    /* vmovupd (not vmovapd): rsp is only guaranteed 16-byte aligned here,
     * not the 32 bytes an aligned YMM store requires -- caused a real
     * segfault before this fix. */
    vmovupd %ymm11, 0(%rsp)      /* acc40: row4, cols0-3 */
    vmovupd %ymm12, 32(%rsp)     /* acc41: row4, cols4-7 */
    vmovupd %ymm13, 64(%rsp)     /* acc50: row5, cols0-3 */
    vmovupd %ymm14, 96(%rsp)     /* acc51: row5, cols4-7 */

    movq    %rdx, %rax           /* rax = C base */
    xorq    %rcx, %rcx           /* rcx = column index 0..7 */
.Lrow45_loop:
    movsd   0(%rsp,%rcx,8), %xmm0   /* row4 value for this column */
    movsd   64(%rsp,%rcx,8), %xmm1  /* row5 value for this column */
    addsd   32(%rax), %xmm0          /* C[col*ldC_bytes + 4*8] */
    addsd   40(%rax), %xmm1          /* C[col*ldC_bytes + 5*8] */
    movsd   %xmm0, 32(%rax)
    movsd   %xmm1, 40(%rax)
    addq    %r8, %rax
    incq    %rcx
    cmpq    $8, %rcx
    jl      .Lrow45_loop

    vzeroupper
    mov     %rbp, %rsp
    pop     %rbp
    ret
    .size kernel_6x8_asm, .-kernel_6x8_asm

    .section .note.GNU-stack,"",@progbits
