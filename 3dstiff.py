from sympy import *

# Assumes the element is a cube: multiplies the final matrix by the volume instead of multiplying each dimension by its size

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

# Rotation around z
theta = symbols('theta')
T = Matrix([[cos(theta)**2, sin(theta)**2, 0, 0, 0, -2*cos(theta)*sin(theta)],
            [sin(theta)**2, cos(theta)**2, 0, 0, 0, 2*cos(theta)*sin(theta)],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, cos(theta), sin(theta), 0],
            [0, 0, 0, -sin(theta), cos(theta), 0],
            [cos(theta)*sin(theta), -cos(theta)*sin(theta), 0, 0, 0, cos(theta)**2-sin(theta)**2]])
Cr = T * C * T.T

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

BCB = B.T * Cr * B
dBCB = diff(BCB,theta)
dK = integrate(integrate(integrate(dBCB, (r,-1,1)), (s,-1,1)), (t,-1,1))

# function code with some hand optimizations to save operations and execute faster
with open('dkdt3d.py','w') as f:
    f.write('import numpy as np\n')
    f.write('def dkdt3d(Ex,Ey,nuxy,nuyz,Gxy,T,V):\n')
    f.write('    c = np.cos(T)\n')
    f.write('    s = np.sin(T)\n')
    f.write('    delta = 1.0*Ex*nuyz**2 - 1.0*Ex + 2.0*Ey*nuxy**2*nuyz + 2.0*Ey*nuxy**2\n')
    f.write('    dkdt = np.zeros((24,24))\n')
    for i in range(24):
        for j in range(i,24):
            line = printing.pycode(dK[i,j]).replace('math.cos(theta)','c').replace('math.sin(theta)','s')
            line = line.replace('(1.0*Ex*nuyz**2 - 1.0*Ex + 2.0*Ey*nuxy**2*nuyz + 2.0*Ey*nuxy**2)','delta')
            
            # searches if expression was already calculated in previous terms
            found = False
            for k in range(i+1):
                if found: break
                for l in range(k,24):
                    if (i,j) == (k,l): break
                    if dK[i,j] == dK[k,l]:
                        line = f'dkdt[{k}][{l}]'
                        found = True
                        break
                    if dK[i,j] == -dK[k,l]:
                        line = f'-dkdt[{k}][{l}]'
                        found = True
                        break

            f.write(f"    dkdt[{i}][{j}] = {line}\n")
    f.write('    dkdt = dkdt + dkdt.T - np.diag(dkdt.diagonal())\n')
    f.write('    dkdt *= V\n')
    f.write('    return dkdt')
