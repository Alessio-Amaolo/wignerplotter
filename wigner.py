# Function that takes a density matrix and plots Q_0(alpha)
# can be imported into other files 
# if this file is run, test with c_nm(0) 

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from linblad import sfactorial

def wigner0(alpha, denmat, cutoff):
    # computes it only for state=|alpha>
    n = np.arange(0, cutoff, 1)
    state = np.exp((-np.abs(alpha)**2)/2)
    temp = [sfactorial(i) for i in n]
    state *= (alpha**n)/temp
    return np.real(np.dot(np.conjugate(state),np.dot(denmat, state))/np.pi)

#def getCnm0(cutoff):
#    beta = 2
#    Cnm = np.zeros((cutoff, cutoff))
#    for n in range(cutoff):
#        for m in range(cutoff):
#            Cnm[n,m] = np.exp(-beta)*(beta**n)*(np.conjugate(beta)**m)/((np.math.factorial(n)*np.math.factorial(m))**(1/2))
#    return Cnm

def wignerGrid(grid, denmat):
    x1, y1 = grid
    nx, ny = y1.shape
    p1 = np.zeros(x1.shape)
    cutoff = denmat.shape[0]
    for i in range(nx):
        for j in range(ny):
            z = complex(x1[i,j],y1[i,j])
            Q0 = wigner0(z, denmat, cutoff)
            p1[i,j] = Q0
    return p1

if __name__ == '__main__':
    #cutoff = 40
    #cnm0 = getCnm0(cutoff)
    cnm0 = np.load('denmat.npy')
    X = np.linspace(-3,3,101)
    grid = np.meshgrid(X, X)
    data = wignerGrid(grid, cnm0)
    fig, ax = plt.subplots(1, figsize=(5,5))
    cax = ax.contourf(X, X, data, levels=100)
    fig.colorbar(cax)
    #ax.set_xticks(X)
    #ax.set_yticks(X)


    plt.show()


