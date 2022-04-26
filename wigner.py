# Function that takes a density matrix and plots Q_0(alpha)
# can be imported into other files 
# if this file is run, test with c_nm(0) 

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

def factorial(arr):
    # make this an element wise factorial!!!
    pass 

def wigner0(alpha, denmat, cutoff):
    # computes it only for state=|alpha>
    n = np.arange(0, cutoff, 1)
    state = np.exp((-np.abs(alpha)**2)/2)
    state *= (alpha**n)*n/((factorial(n))**(1/2))
    return np.dot(np.conjugate(state),np.dot(denmat, state))/np.pi

def getCnm0(cutoff):
    beta = 4
    Cnm = np.zeros((cutoff, cutoff))
    for n in range(cutoff):
        for m in range(cutoff):
            Cnm[n,m] = np.exp(-beta)*(beta**n)*(np.conjugate(beta)**m)/((np.math.factorial(n)*np.math.factorial(m))**(1/2))
    return Cnm

def wignerGrid(grid, denmat):
    x1, y1 = grid
    nx, ny = y1.shape
    p1 = x1*0
    cutoff = denmat.shape[0]
    for i in range(nx):
        for j in range(ny):
            z = complex(x1[i,j],y1[i,j])
            Q0 = wigner0(z, denmat, cutoff)
            p1[i,j] = Q0
    return p1

if __name__ == '__main__':
    cutoff = 40
    cnm0 = getCnm0(cutoff)
    grid = np.meshgrid(np.linspace(-2.5,2.5,101), np.linspace(-2.5,2.5,101))
    data = wignerGrid(grid, cnm0)
    plt.pcolormech(data)
    plt.show()


