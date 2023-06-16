from sympy import *

V = symbols('V')

# Transverse isotropy, symmetry planne yz
# Constants: Ex, Ey, nuxy, nuyz, Gxy
# Non-independent:
# Ez = Ey
# nuxz = nuxy
# Gzx = Gxy
# Gyz = Ey/(2*(1+nuyz))
Ex, Ey, nuxy, nuyz, Gxy = symbols('Ex Ey nuxy nuyz Gxy')
nuyx = nuxy*Ey/Ex

S = Matrix([[1/Ex,-nuxy/Ey,-nuxy/Ey,0,0,0],[-nuyx/Ex,1/Ey,-nuyz/Ey,0,0,0],[-nuyx/Ex,-nuyz/Ey,1/Ey,0,0,0],[0,0,0,2*(1+nuyz)/Ey,0,0],[0,0,0,0,1/Gxy,0],[0,0,0,0,0,1/Gxy]])
C = S.inv()

theta = symbols('theta')
R = Matrix([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])

rot = R * C[[0,1,5],[0,1,5]] * R.T
C[0,0] = rot[0,0]
C[0,1] = rot[0,1]
C[0,5] = rot[0,2]
C[1,0] = rot[1,0]
C[1,1] = rot[1,1]
C[1,5] = rot[1,2]
C[5,0] = rot[2,0]
C[5,1] = rot[2,1]
C[5,5] = rot[2,2]

r, s, t = symbols('r s t')

a = 1/8 * (1-r) * (1-s) * (1-t)
b = 1/8 * (1+r) * (1-s) * (1-t)
c = 1/8 * (1+r) * (1+s) * (1-t)
d = 1/8 * (1-r) * (1+s) * (1-t)
e = 1/8 * (1-r) * (1-s) * (1+t)
f = 1/8 * (1+r) * (1-s) * (1+t)
g = 1/8 * (1+r) * (1+s) * (1+t)
h = 1/8 * (1-r) * (1+s) * (1+t)

B1 = Matrix([[diff(a,r), 0, 0], [0, diff(a,s), 0], [0, 0, diff(a,t)], [0, diff(a,t), diff(a,s)], [diff(a,t), 0, diff(a,r)], [diff(a,s), diff(a,r), 0]])
B2 = Matrix([[diff(b,r), 0, 0], [0, diff(b,s), 0], [0, 0, diff(b,t)], [0, diff(b,t), diff(b,s)], [diff(b,t), 0, diff(b,r)], [diff(b,s), diff(b,r), 0]])
B3 = Matrix([[diff(c,r), 0, 0], [0, diff(c,s), 0], [0, 0, diff(c,t)], [0, diff(c,t), diff(c,s)], [diff(c,t), 0, diff(c,r)], [diff(c,s), diff(c,r), 0]])
B4 = Matrix([[diff(d,r), 0, 0], [0, diff(d,s), 0], [0, 0, diff(d,t)], [0, diff(d,t), diff(d,s)], [diff(d,t), 0, diff(d,r)], [diff(d,s), diff(d,r), 0]])
B5 = Matrix([[diff(e,r), 0, 0], [0, diff(e,s), 0], [0, 0, diff(e,t)], [0, diff(e,t), diff(e,s)], [diff(e,t), 0, diff(e,r)], [diff(e,s), diff(e,r), 0]])
B6 = Matrix([[diff(f,r), 0, 0], [0, diff(f,s), 0], [0, 0, diff(f,t)], [0, diff(f,t), diff(f,s)], [diff(f,t), 0, diff(f,r)], [diff(f,s), diff(f,r), 0]])
B7 = Matrix([[diff(g,r), 0, 0], [0, diff(g,s), 0], [0, 0, diff(g,t)], [0, diff(g,t), diff(g,s)], [diff(g,t), 0, diff(g,r)], [diff(g,s), diff(g,r), 0]])
B8 = Matrix([[diff(h,r), 0, 0], [0, diff(h,s), 0], [0, 0, diff(h,t)], [0, diff(h,t), diff(h,s)], [diff(h,t), 0, diff(h,r)], [diff(h,s), diff(h,r), 0]])

B = Matrix([[B1, B2, B3, B4, B5, B6, B7, B8]])

BC = V * B.T * C * B
K = integrate(integrate(integrate(BC, (r,-1,1)), (s,-1,1)), (t,-1,1))
dK = diff(K,theta)

code = printing.pycode(dK)
with open('sensitivity.txt','w') as f:
    f.write(code)
