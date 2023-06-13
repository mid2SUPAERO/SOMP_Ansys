import numpy as np
def dkdt2d(Ex,Ey,nuxy,T,V):
    dkdt = np.zeros((8,8))
    dkdt[0,0] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[0,1] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[0,2] = ((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[0,3] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[0,4] = ((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[0,5] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[0,6] = -((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[0,7] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[1,1] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
    dkdt[1,2] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[1,3] = (4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
    dkdt[1,4] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[1,5] = -(4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
    dkdt[1,6] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[1,7] = -(4*Ex**2*np.cos(T)*np.sin(T) - 4*Ex*Ey*np.cos(T)*np.sin(T) + 4*Ex*Ey*nuxy*np.cos(T)**2 - 4*Ex*Ey*nuxy*np.sin(T)**2)/(- 6*Ey*nuxy**2 + 6*Ex)
    dkdt[2,2] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[2,3] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[2,4] = -((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[2,5] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[2,6] = ((Ex**2*np.cos(T)*np.sin(T))/3 - (Ex*Ey*np.cos(T)*np.sin(T))/3 + (Ex*Ey*nuxy*np.cos(T)**2)/3 - (Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[2,7] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[3,3] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
    dkdt[3,4] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[3,5] = -(4*Ex**2*np.cos(T)*np.sin(T) - 4*Ex*Ey*np.cos(T)*np.sin(T) + 4*Ex*Ey*nuxy*np.cos(T)**2 - 4*Ex*Ey*nuxy*np.sin(T)**2)/(- 6*Ey*nuxy**2 + 6*Ex)
    dkdt[3,6] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[3,7] = -(4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
    dkdt[4,4] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[4,5] = -((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[4,6] = ((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[4,7] = -(Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[5,5] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
    dkdt[5,6] = (Ex*Ey*np.cos(2*T) - Ex**2*np.cos(2*T) + 2*Ex*Ey*nuxy*np.sin(2*T))/(4*(- Ey*nuxy**2 + Ex))
    dkdt[5,7] = (4*((Ex**2*np.cos(T)*np.sin(T))/4 - (Ex*Ey*np.cos(T)*np.sin(T))/4 + (Ex*Ey*nuxy*np.cos(T)**2)/4 - (Ex*Ey*nuxy*np.sin(T)**2)/4))/(3*(- Ey*nuxy**2 + Ex))
    dkdt[6,6] = -((2*Ex**2*np.cos(T)*np.sin(T))/3 - (2*Ex*Ey*np.cos(T)*np.sin(T))/3 + (2*Ex*Ey*nuxy*np.cos(T)**2)/3 - (2*Ex*Ey*nuxy*np.sin(T)**2)/3)/(- Ey*nuxy**2 + Ex)
    dkdt[6,7] = ((Ex*Ey*np.cos(2*T))/4 - (Ex**2*np.cos(2*T))/4 + (Ex*Ey*nuxy*np.sin(2*T))/2)/(- Ey*nuxy**2 + Ex)
    dkdt[7,7] = (2*Ex**2*np.cos(T)*np.sin(T) - 2*Ex*Ey*np.cos(T)*np.sin(T) + 2*Ex*Ey*nuxy*np.cos(T)**2 - 2*Ex*Ey*nuxy*np.sin(T)**2)/(- 3*Ey*nuxy**2 + 3*Ex)
    dkdt = dkdt + dkdt.T - np.diag(dkdt.diagonal()) # symmetric matrix
    dkdt = dkdt * V
        
    return dkdt
