import numpy as np
from scipy.sparse import coo_matrix, diags, identity

"""
Weights: max{0, rmin-distance(i,j)}
"""
class ConvolutionFilter():
    def __init__(self, rmin, centers):
        num_elem = centers.shape[0]
        self.H = ConvolutionFilter.preFlt(rmin, num_elem, centers)
        
    def distances(matrixA, matrixB):
        A = np.matrix(matrixA)
        B = np.matrix(matrixB)
        Btrans = B.transpose()
        vecProd = A * Btrans
        SqA =  A.getA()**2
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
        SqB = B.getA()**2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
        SqED = sumSqBEx + sumSqAEx - 2*vecProd   
        elmDis = (np.maximum(0,SqED).getA())**0.5
        return np.matrix(elmDis)
    
    # https://www.rmit.edu.au/research/centres-collaborations/centre-for-innovative-structures-and-materials/software
    def preFlt(rmin, num_elem, centers):
        if rmin == 0: return identity(num_elem)
        limitElementNumber = 4000 # should be larger than (2*(rmin/elmsize)) ** 3
        nfilter=int(num_elem*limitElementNumber) 
        iH,jH,sH,cc = np.zeros(nfilter),np.zeros(nfilter),np.zeros(nfilter),0
        for ei in range(num_elem):
            ii = np.where(abs(centers[:,0] - centers[ei][0]) < rmin)[0]
            jj = np.where(abs(centers[ii,1] - centers[ei][1]) < rmin)[0]
            kk = np.where(abs(centers[ii[jj],2] - centers[ei][2]) < rmin)[0]
            neighbors = ii[jj][kk]
            iH[cc:cc+len(neighbors)] = ei
            jH[cc:cc+len(neighbors)] = neighbors
            eiH = np.maximum(0,rmin - ConvolutionFilter.distances(centers[ei],centers[neighbors]))
            sH[cc:cc+len(neighbors)] = eiH / np.sum(eiH)
            cc += len(neighbors)
        H = coo_matrix((sH,(iH,jH)),shape=(num_elem,num_elem)).tocsr()
        return H
    
class DensityFilter(ConvolutionFilter):
    def filter(self, rho, dc):
        return self.H.dot(rho*dc)/rho

class OrientationFilter(ConvolutionFilter):    
    def filter(self, rho, theta, alpha):
        # ignore element in filter if density is near to blank
        # multiply each column by the correspondent rho
        a = diags(rho)
        H = self.H @ a
        
        # divide each line by its sum (renormalize weights)
        a = diags(1/H.sum(axis=1).A.ravel())
        H = a @ H
        
        return H.dot(theta), H.dot(alpha)
