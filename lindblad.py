import numpy as np
from numba import njit, prange, set_num_threads

@njit
def sfactorial(i):
    if i == 0:
        return 1
    else:
        return np.prod(np.sqrt(np.arange(1,i+1,1)))

@njit
def ic(i, j, beta):
    num = np.exp(-np.abs(beta)**2)*(beta**(i+j))
    num /= sfactorial(i)
    num /= sfactorial(j)
    return np.exp(- np.abs(beta) ** 2) * beta ** (i + j) / (sfactorial(i)*sfactorial(j))

@njit
def evolve(N, tf, nt, numsteps, steps_per_anim, beta, kappa, K):
    dt = tf/nt
    rho = np.zeros((numsteps,N,N), dtype = np.complex64)
    curr_rho = np.zeros((N,N), dtype = np.complex64)
    idx = 0
    for i in range(N):
        for j in range(N):
            curr_rho[i,j] = ic(i,j, beta)
    # make imask and jmask, can pre-multiply i*i-j*j, etc
    # Then use the mask properties to make sure N-1 boundaries are respected
    # Finally, evaluate the derivative and compute curr_rho quickly 
    for t_i in range(nt):
        if t_i % steps_per_anim == 0:
            rho[idx,:,:] = curr_rho
            idx += 1

        for i in range(N):
            for j in range(N):
                derivative = curr_rho[i, j]* (1j*(K/2)*(i*(i)-j*(j)) - kappa*(i+j)/2)
                if i < N-1 and j < N-1:
                    derivative += curr_rho[i+1,j+1]*kappa*(np.sqrt(i*j))
                curr_rho[i, j] = curr_rho[i, j] + derivative * dt
        if t_i % 100000 == 0:
            print(t_i)
    return rho

if __name__ == "__main__":
    N = 35
    beta = 2
    numsteps = 3066
    steps_per_anim = 1000
    tf = (10 ** (-9))*numsteps
    nt = numsteps*steps_per_anim
    #nt = (10 ** 6)
    dt = (tf / nt)
    # resolution = nt//numsteps
    kappa = 0 # 2 * np.pi * 10 * 10 ** 3
    K = 2 * np.pi * 325 * 10 ** 3
    rho = evolve(N, tf, nt, numsteps, steps_per_anim, beta, kappa, K)
    np.save(f"DensityMatrixEvolved_numsteps={numsteps}_N={N}_k={kappa}.npy", rho)

