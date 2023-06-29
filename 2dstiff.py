from sympy import *

Ex, Ey, nuxy, Gxy = symbols('Ex Ey nuxy Gxy')
nuyx = nuxy*Ey/Ex

S = Matrix([[1/Ex,-nuxy/Ey,0],[-nuyx/Ex,1/Ey,0],[0,0,1/Gxy]])
C = S.inv()

theta = symbols('theta')
T = Matrix([[cos(theta)**2, sin(theta)**2, -2*cos(theta)*sin(theta)],
            [sin(theta)**2, cos(theta)**2, 2*cos(theta)*sin(theta)],
            [cos(theta)*sin(theta), -cos(theta)*sin(theta), cos(theta)**2-sin(theta)**2]])
Cr = T * C * T.T

r, s = symbols('r s')
a = 1/4 * (1-r) * (1-s)
b = 1/4 * (1+r) * (1-s)
c = 1/4 * (1+r) * (1+s)
d = 1/4 * (1-r) * (1+s)

B1 = Matrix([[diff(a,r), 0], [0, diff(a,s)], [diff(a,s), diff(a,r)]])
B2 = Matrix([[diff(b,r), 0], [0, diff(b,s)], [diff(b,s), diff(b,r)]])
B3 = Matrix([[diff(c,r), 0], [0, diff(c,s)], [diff(c,s), diff(c,r)]])
B4 = Matrix([[diff(d,r), 0], [0, diff(d,s)], [diff(d,s), diff(d,r)]])

B = Matrix([[B1, B2, B3, B4]])

BCB = B.T * Cr * B
dBCB = diff(BCB, theta)
dK = integrate(integrate(dBCB, (r,-1,1)), (s,-1,1))

# function code with some hand optimizations to save operations and execute faster
with open('dkdt2d.py','w') as f:
    f.write('import numpy as np\n')
    f.write('def dkdt2d(Ex,Ey,nuxy,nuyz,Gxy,T,V):\n')
    f.write('    c = np.cos(T)\n')
    f.write('    s = np.sin(T)\n')
    f.write('    delta = 1.0*Ex - 1.0*Ey*nuxy**2\n')
    f.write('    dkdt = np.zeros((8,8))\n')
    for i in range(8):
        for j in range(i,8):
            line = printing.pycode(dK[i,j]).replace('math.cos(theta)','c').replace('math.sin(theta)','s')
            line = line.replace('(1.0*Ex - 1.0*Ey*nuxy**2)', 'delta')
            
            # searches if expression was already calculated in previous terms
            found = False
            for k in range(i+1):
                if found: break
                for l in range(k,8):
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
    f.write('    dkdt += dkdt.T - np.diag(dkdt.diagonal())\n')
    f.write('    dkdt *= V\n')
    f.write('    return dkdt')
