# Function that takes a density matrix and plots Q_0(alpha)
# can be imported into other files 
# if this file is run, test with c_nm(0) 

import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
from linblad import sfactorial
import time

@njit
def getAlphaState(alpha, cutoff):
    prefact = np.exp((-np.abs(alpha)**2)/2)
    state = np.zeros(cutoff, dtype=np.complex64)
    for n in range(cutoff):
        state[n] = alpha**n/sfactorial(n)
    state = state*prefact
    return state

@jit(nopython=False)
def wigner0(alpha, denmat, cutoff):
    # computes it only for state=|alpha>
    state = getAlphaState(alpha, cutoff)
    return np.real(np.dot(np.conjugate(state),np.dot(denmat, state))/np.pi)

@jit(nopython=False)
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
    N = 35
    beta = 2
    numsteps = 3066
    denmat = np.load(f"DensityMatrixEvolved_numsteps={numsteps}_N={N}.npy")

    X = np.linspace(-3,3,101)
    grid = np.meshgrid(X, X)
    tlist = np.arange(0, numsteps, 4)
    extra = np.array([15, 65, 440, 815, 1065, 1565, 2565, 3065])
    As = [1, 0.93, 0.32, 0.24, 0.3, 0.46, 0.24, 0.89]
    extraidx = 0
    tlist = np.append(tlist, extra)
    for t in tlist:
        data = wignerGrid(grid, denmat[t,:,:])
        fig, ax = plt.subplots(1, figsize=(6,5))
        ax.set_title(f"$t=${t}ns")
        ax.set_xlabel(r"$\Re (\alpha)$")
        ax.set_ylabel(r"$\Im (\alpha)$")
        name = "t="+f"{t}".zfill(4)

        if np.any(extra == t):
            A = As[extraidx]
            cax = ax.contourf(X, X, data, levels=100) #, vmin=-0.1/np.pi, vmax=A/np.pi)
            cbar = fig.colorbar(cax)
            #cbar.set_ticks([-0.1/np.pi, 0, A/np.pi])
            #cbar.set_ticklabels([r'$\frac{-0.1}{\pi}$', 0, r'$\frac{A}{\pi}$'])
            plt.savefig(f"SpecialFig3/{name}.png", bbox_inches='tight')
            extraidx += 1
        else:
            cax = ax.contourf(X, X, data, levels=100)
            fig.colorbar(cax)
            plt.savefig(f"data/{name}.png", bbox_inches='tight')

        print(f"Done with t={t}")

        plt.clf()
        plt.close()


