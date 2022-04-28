import numpy as np
import matplotlib.pyplot as plt 
from numba import njit 

@njit
def compute_sum(alpha, beta, K, upper, t):
    denom = 1
    num = np.conjugate(alpha)*beta
    total = 0
    term = num/denom
    for n in range(1, upper+1):
        term /= n
        total += term*np.exp(1j*K*(n**2)*t/2)
        term *= num
    return total

@njit
def evolution(grid, timesteps, interval, savinterval):
    x1, y1 = grid
    nx, ny = y1.shape
    p1 = x1*0
    beta = 2
    K = 2*np.pi*325*(10**3)
    prefact = np.exp((-np.abs(beta)**2)/2)
    savesteps = timesteps//savinterval+1
    p1_intime = np.zeros((p1.shape[0], p1.shape[1], savesteps))
    tcounter = 0
    for t in range(0, timesteps, interval):
        for i in range(nx):
            for j in range(ny):
                z = complex(x1[i,j],y1[i,j])
                total = compute_sum(z, beta, K, 50, t*(10**-9))
                value = (np.abs((prefact*np.exp(-np.abs(z)**2)/2)*(1+total))**2)/np.pi
                p1[i,j] = value
        if t%savinterval == 0:
            print(t, tcounter)
            p1_intime[:,:, tcounter] = p1
            tcounter += 1
    p1_intime[:,:,-1] = p1
    
    return p1_intime
    #plt.figure(figsize=(10,10))
    #plt.contour(x1, y1, p1)
    #plt.savefig(f"1a.png", bbox_inches='tight')

if __name__ == '__main__':
    X = np.linspace(-3, 3, 101)
    grid = np.meshgrid(X, X)
    savinterval = 1
    timesteps = 1 #3060
    data = evolution(grid, timesteps, 5, savinterval)

    numimages = timesteps//savinterval+1
    for i in range(numimages):
        plt.figure(figsize=(6,6))
        plt.contourf(X, X, data[:,:,i])
        plt.title(f"$t=${i*savinterval} ns")
        name = "t="+f"{i*savinterval}".zfill(4)
        plt.savefig(f'exactEvol/{name}', bbox_inches='tight')
        plt.clf()
        plt.close()

        print(f"Done with {name}")
    


    
