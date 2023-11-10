import numpy as np

def dk2d(Ex,Ey,nuxy,nuyz,Gxy,T,A,V):
    points  = [-1/np.sqrt(3), 1/np.sqrt(3)]
    weights = [1, 1]
    
    C   = C2d(Ex,Ey,nuxy,nuyz,Gxy)
    Tt  = Tt_fun(T)[[0,1,5],:][:,[0,1,5]]
    dTt = dTt_fun(T)[[0,1,5],:][:,[0,1,5]]
    
    dkdt = np.zeros((8,8))
    for xi, wi in zip(points, weights):
        for xj, wj in zip(points, weights):
                B = B2d(xi, xj)
                dkdt += wi * wj * B.T @ (dTt @ C @ Tt.T + Tt @ C @ dTt.T) @ B
    
    dkdt *= V
    return dkdt, np.zeros((8,8))

def dk3d(Ex,Ey,nuxy,nuyz,Gxy,T,A,V,print_euler):
    points  = [-1/np.sqrt(3), 1/np.sqrt(3)]
    weights = [1, 1]
    
    C   = C3d(Ex,Ey,nuxy,nuyz,Gxy)
    Ta  = Ta_fun(A)
    dTa = dTa_fun(A)
    Tt  = Tt_fun(T)
    dTt = dTt_fun(T)
    
    # C in the printing coordinate system
    euler1, euler2 = print_euler
    Tprint = Ta_fun(euler2) @ Tt_fun(euler1)
    C = Tprint @ C @ Tprint.T
    
    dkdt, dkda = np.zeros((24,24)), np.zeros((24,24))
    for xi, wi in zip(points, weights):
        for xj, wj in zip(points, weights):
            for xk, wk in zip(points, weights):
                B = B3d(xi, xj, xk)
                dkdt += wi * wj * wk * B.T @ Ta @ (dTt @ C @ Tt.T + Tt @ C @ dTt.T) @ Ta.T @ B
                dkda += wi * wj * wk * B.T @ (dTa @ Tt @ C @ Tt.T @ Ta.T + Ta @ Tt @ C @Tt.T @ dTa.T) @ B
    
    dkdt *= V
    dkda *= V
    return dkdt, dkda

def C2d(Ex,Ey,nuxy,nuyz,Gxy):
    C = np.zeros((3,3))
    C[0][0] = Ex**2/(Ex - Ey*nuxy**2)
    C[0][1] = (Ex*Ey*nuxy)/(Ex - Ey*nuxy**2)
    C[0][2] = 0
    C[1][1] = (Ex*Ey)/(Ex - Ey*nuxy**2)
    C[1][2] = 0
    C[2][2] = Gxy
    C += C.T - np.diag(C.diagonal())
    return C

def C3d(Ex,Ey,nuxy,nuyz,Gxy):
    C = np.zeros((6,6))
    C[0][0] = (Ex**2*nuyz - Ex**2)/(Ex*nuyz - Ex + 2*Ey*nuxy**2)
    C[0][1] = -(Ex*Ey*nuxy)/(Ex*nuyz - Ex + 2*Ey*nuxy**2)
    C[0][2] = -(Ex*Ey*nuxy)/(Ex*nuyz - Ex + 2*Ey*nuxy**2)
    C[0][3] = 0
    C[0][4] = 0
    C[0][5] = 0
    C[1][1] = -(Ey*(Ex - Ey*nuxy**2))/(Ex*nuyz**2 - Ex + 2*Ey*nuxy**2 + 2*Ey*nuxy**2*nuyz)
    C[1][2] = -(Ey*(Ex*nuyz + Ey*nuxy**2))/(Ex*nuyz**2 - Ex + 2*Ey*nuxy**2 + 2*Ey*nuxy**2*nuyz)
    C[1][3] = 0
    C[1][4] = 0
    C[1][5] = 0
    C[2][2] = -(Ey*(Ex - Ey*nuxy**2))/(Ex*nuyz**2 - Ex + 2*Ey*nuxy**2 + 2*Ey*nuxy**2*nuyz)
    C[2][3] = 0
    C[2][4] = 0
    C[2][5] = 0
    C[3][3] = Ey/(2*(nuyz + 1))
    C[3][4] = 0
    C[3][5] = 0
    C[4][4] = Gxy
    C[4][5] = 0
    C[5][5] = Gxy
    C += C.T - np.diag(C.diagonal())
    return C

def Ta_fun(A):
    c  = np.cos(A)
    s  = np.sin(A)
    Ta = np.zeros((6,6))
    Ta[0][0] = 1
    Ta[1][1] = c**2
    Ta[1][2] = s**2
    Ta[1][3] = -2*c*s
    Ta[2][1] = s**2
    Ta[2][2] = c**2
    Ta[2][3] = 2*c*s
    Ta[3][1] = c*s
    Ta[3][2] = -c*s
    Ta[3][3] = c**2 - s**2
    Ta[4][4] = c
    Ta[4][5] = s
    Ta[5][4] = -s
    Ta[5][5] = c
    return Ta

def dTa_fun(A):
    c  = np.cos(A)
    s  = np.sin(A)
    dTa = np.zeros((6,6))
    dTa[1][1] = -2*c*s
    dTa[1][2] = 2*c*s
    dTa[1][3] = 2*s**2 - 2*c**2
    dTa[2][1] = 2*c*s
    dTa[2][2] = -2*c*s
    dTa[2][3] = 2*c**2 - 2*s**2
    dTa[3][1] = c**2 - s**2
    dTa[3][2] = s**2 - c**2
    dTa[3][3] = -4*c*s
    dTa[4][4] = -s
    dTa[4][5] = c
    dTa[5][4] = -c
    dTa[5][5] = -s
    return dTa

def Tt_fun(T):
    c  = np.cos(T)
    s  = np.sin(T)
    Tt = np.zeros((6,6))
    Tt[0][0] = c**2
    Tt[0][1] = s**2
    Tt[0][5] = -2*c*s
    Tt[1][0] = s**2
    Tt[1][1] = c**2
    Tt[1][5] = 2*c*s
    Tt[2][2] = 1
    Tt[3][3] = c
    Tt[3][4] = s
    Tt[4][3] = -s
    Tt[4][4] = c
    Tt[5][0] = c*s
    Tt[5][1] = -c*s
    Tt[5][5] = c**2 - s**2
    return Tt

def dTt_fun(T):
    c  = np.cos(T)
    s  = np.sin(T)
    dTt = np.zeros((6,6))
    dTt[0][0] = -2*c*s
    dTt[0][1] = 2*c*s
    dTt[0][5] = 2*s**2 - 2*c**2
    dTt[1][0] = 2*c*s
    dTt[1][1] = -2*c*s
    dTt[1][5] = 2*c**2 - 2*s**2
    dTt[3][3] = -s
    dTt[3][4] = c
    dTt[4][3] = -c
    dTt[4][4] = -s
    dTt[5][0] = c**2 - s**2
    dTt[5][1] = s**2 - c**2
    dTt[5][5] = -4*c*s
    return dTt

def B2d(r,s):
    B = np.zeros((3,8))
    B[0][0] = s/4 - 1/4
    B[0][2] = 1/4 - s/4
    B[0][4] = s/4 + 1/4
    B[0][6] = - s/4 - 1/4
    B[1][1] = r/4 - 1/4
    B[1][3] = - r/4 - 1/4
    B[1][5] = r/4 + 1/4
    B[1][7] = 1/4 - r/4
    B[2][0] = r/4 - 1/4
    B[2][1] = s/4 - 1/4
    B[2][2] = - r/4 - 1/4
    B[2][3] = 1/4 - s/4
    B[2][4] = r/4 + 1/4
    B[2][5] = s/4 + 1/4
    B[2][6] = 1/4 - r/4
    B[2][7] = - s/4 - 1/4
    return B

def B3d(r,s,t):
    B = np.zeros((6,24))
    B[0][0] = -((s - 1)*(t - 1))/8
    B[0][3] = ((s - 1)*(t - 1))/8
    B[0][6] = -((s + 1)*(t - 1))/8
    B[0][9] = ((s + 1)*(t - 1))/8
    B[0][12] = ((s - 1)*(t + 1))/8
    B[0][15] = -((s - 1)*(t + 1))/8
    B[0][18] = ((s + 1)*(t + 1))/8
    B[0][21] = -((s + 1)*(t + 1))/8
    B[1][1] = -(r/8 - 1/8)*(t - 1)
    B[1][4] = (r/8 + 1/8)*(t - 1)
    B[1][7] = -(r/8 + 1/8)*(t - 1)
    B[1][10] = (r/8 - 1/8)*(t - 1)
    B[1][13] = (r/8 - 1/8)*(t + 1)
    B[1][16] = -(r/8 + 1/8)*(t + 1)
    B[1][19] = (r/8 + 1/8)*(t + 1)
    B[1][22] = -(r/8 - 1/8)*(t + 1)
    B[2][2] = -(r/8 - 1/8)*(s - 1)
    B[2][5] = (r/8 + 1/8)*(s - 1)
    B[2][8] = -(r/8 + 1/8)*(s + 1)
    B[2][11] = (r/8 - 1/8)*(s + 1)
    B[2][14] = (r/8 - 1/8)*(s - 1)
    B[2][17] = -(r/8 + 1/8)*(s - 1)
    B[2][20] = (r/8 + 1/8)*(s + 1)
    B[2][23] = -(r/8 - 1/8)*(s + 1)
    B[3][1] = -(r/8 - 1/8)*(s - 1)
    B[3][2] = -(r/8 - 1/8)*(t - 1)
    B[3][4] = (r/8 + 1/8)*(s - 1)
    B[3][5] = (r/8 + 1/8)*(t - 1)
    B[3][7] = -(r/8 + 1/8)*(s + 1)
    B[3][8] = -(r/8 + 1/8)*(t - 1)
    B[3][10] = (r/8 - 1/8)*(s + 1)
    B[3][11] = (r/8 - 1/8)*(t - 1)
    B[3][13] = (r/8 - 1/8)*(s - 1)
    B[3][14] = (r/8 - 1/8)*(t + 1)
    B[3][16] = -(r/8 + 1/8)*(s - 1)
    B[3][17] = -(r/8 + 1/8)*(t + 1)
    B[3][19] = (r/8 + 1/8)*(s + 1)
    B[3][20] = (r/8 + 1/8)*(t + 1)
    B[3][22] = -(r/8 - 1/8)*(s + 1)
    B[3][23] = -(r/8 - 1/8)*(t + 1)
    B[4][0] = -(r/8 - 1/8)*(s - 1)
    B[4][2] = -((s - 1)*(t - 1))/8
    B[4][3] = (r/8 + 1/8)*(s - 1)
    B[4][5] = ((s - 1)*(t - 1))/8
    B[4][6] = -(r/8 + 1/8)*(s + 1)
    B[4][8] = -((s + 1)*(t - 1))/8
    B[4][9] = (r/8 - 1/8)*(s + 1)
    B[4][11] = ((s + 1)*(t - 1))/8
    B[4][12] = (r/8 - 1/8)*(s - 1)
    B[4][14] = ((s - 1)*(t + 1))/8
    B[4][15] = -(r/8 + 1/8)*(s - 1)
    B[4][17] = -((s - 1)*(t + 1))/8
    B[4][18] = (r/8 + 1/8)*(s + 1)
    B[4][20] = ((s + 1)*(t + 1))/8
    B[4][21] = -(r/8 - 1/8)*(s + 1)
    B[4][23] = -((s + 1)*(t + 1))/8
    B[5][0] = -(r/8 - 1/8)*(t - 1)
    B[5][1] = -((s - 1)*(t - 1))/8
    B[5][3] = (r/8 + 1/8)*(t - 1)
    B[5][4] = ((s - 1)*(t - 1))/8
    B[5][6] = -(r/8 + 1/8)*(t - 1)
    B[5][7] = -((s + 1)*(t - 1))/8
    B[5][9] = (r/8 - 1/8)*(t - 1)
    B[5][10] = ((s + 1)*(t - 1))/8
    B[5][12] = (r/8 - 1/8)*(t + 1)
    B[5][13] = ((s - 1)*(t + 1))/8
    B[5][15] = -(r/8 + 1/8)*(t + 1)
    B[5][16] = -((s - 1)*(t + 1))/8
    B[5][18] = (r/8 + 1/8)*(t + 1)
    B[5][19] = ((s + 1)*(t + 1))/8
    B[5][21] = -(r/8 - 1/8)*(t + 1)
    B[5][22] = -((s + 1)*(t + 1))/8
    return B