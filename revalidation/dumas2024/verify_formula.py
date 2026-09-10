"""
Symbolic + numeric verification of Dumas, Pernet & Sedoglavic's 2024
alternate-basis 2x2 matrix-multiplication algorithm ("Strassen's
algorithm is not optimally accurate", ISSAC 2024, arXiv:2402.05630,
Equation 36 / Table 3), transcribed from the paper's straight-line
program, adjusted to standard row-major A[i][j]/B[i][j] indexing.

Run before writing avx2/ladder/dumas_strassen_rebuild.c, to catch a
transcription error (the paper's a_ij/b_ij subscript convention turned
out to put a12 where row-major indexing puts a21, and vice versa --
this script is what caught that, by brute-force searching input/output
relabelings and finding the exact permutation that makes the formula
equal standard C = A*B) BEFORE it could silently corrupt the accuracy
study downstream. Re-run this (`python3 verify_formula.py`) if
dumas_strassen_rebuild.c's base-case formula is ever edited -- it
should print four zero diffs and a numeric spot check that matches to
float precision.

Requires: sympy (pip install sympy; version 1.13.2 used originally).
"""
import sympy as sp

A11, A12, A21, A22, B11, B12, B21, B22 = sp.symbols('A11 A12 A21 A22 B11 B12 B21 B22')
sqrt3 = sp.sqrt(3)

# Table 3's slp (24 add., 12 mul./div.), with the paper's a12<->a21 /
# b12<->b21 subscript convention already corrected to row-major.
t1 = (sqrt3/3)*A22
t2 = A12 + t1
t3 = A21 + t2

s1 = (sqrt3/3)*B12
s2 = s1 - B11
s3 = s2 + B22

l1 = (sqrt3/2)*A11 + sp.Rational(1, 2)*t3
l2 = A21 - t1
l3 = t2
l4 = 2*t1
l5 = l2 - l1
l6 = l5 + l4
l7 = l5 + l3

r1 = 2*s1
r2 = s2
r3 = s1 - B22
r4 = sp.Rational(1, 2)*s3 - (sqrt3/2)*B21
r5 = r3 + r4
r6 = r1 - r5
r7 = r5 - r2

p1 = sp.expand(l1*r1)
p2 = sp.expand(l2*r2)
p3 = sp.expand(l3*r3)
p4 = sp.expand(l4*r4)
p5 = sp.expand(l5*r5)
p6 = sp.expand(l6*r6)
p7 = sp.expand(l7*r7)

w2 = p5 + p1 + p6
w1 = p7 + p6
w3 = w2 - p2
w5 = (p4 + w2)/2

C12 = p1 - p3 - w5
C21 = w3 - w5
C22 = sqrt3*w5
C11 = (sqrt3/3)*(w3 - C12 - 2*w1)

C11 = sp.simplify(sp.expand(C11))
C12 = sp.simplify(sp.expand(C12))
C21 = sp.simplify(sp.expand(C21))
C22 = sp.simplify(sp.expand(C22))

ref11 = A11*B11 + A12*B21
ref12 = A11*B12 + A12*B22
ref21 = A21*B11 + A22*B21
ref22 = A21*B12 + A22*B22

print("C11 diff:", sp.simplify(C11 - ref11))
print("C12 diff:", sp.simplify(C12 - ref12))
print("C21 diff:", sp.simplify(C21 - ref21))
print("C22 diff:", sp.simplify(C22 - ref22))

print()
print("Numeric spot check (formula value, reference value):")
import random
random.seed(0)
subs = {s: random.uniform(-5, 5) for s in [A11, A12, A21, A22, B11, B12, B21, B22]}
print(float(C11.subs(subs)), float(ref11.subs(subs)))
print(float(C12.subs(subs)), float(ref12.subs(subs)))
print(float(C21.subs(subs)), float(ref21.subs(subs)))
print(float(C22.subs(subs)), float(ref22.subs(subs)))
