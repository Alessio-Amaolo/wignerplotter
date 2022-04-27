import numpy as np
from numba import njit, prange, set_num_threads

@njit
def factorial(i):
	if i == 0:
		return 1
	else:
		r = 1
		while i > 1:
			r *= i
			i -= 1
		return r
@njit
def ic(i, j, beta):
	return np.exp(- np.abs(beta) ** 2) * beta ** (i + j) / (factorial(i)**(1/2)*factorial(j)**(1/2))

@njit(parallel=True)
def evolve(N, tf, nt, beta, kappa, K):	
	dt = tf/nt
	rho = np.zeros((nt,N,N), dtype = np.complex128)
	for i in prange(N):
		for j in range(N):
			# assuming beta is real (which it is)
			#denom = (factorial(i))**(1/2) * (factorial(j))**(1/2)
			rho[0, i, j] = np.exp(- np.abs(beta) ** 2) * beta ** (i + j) / (factorial(i)**(1/2)*factorial(j)**(1/2))
	print(rho)
	for t_i in range(nt-1):
		for i in prange(N):
			for j in range(N):
				derivative = rho[t_i,i, j]* ((K/2)*(i*(i-1)-j*(j-1)) + kappa*(i+j)/2)
				if i < N-1 and j < N-1:
					derivative += rho[t_i,i+1,j+1]*kappa*(np.sqrt(i*j))
				rho[t_i+1, i, j] = rho[t_i, i, j] + derivative * dt
	return rho

if __name__ == "__main__":
	set_num_threads(8)
	N = 50
	beta = 2
	tf = 3065 * 10 ** (-9)
	nt = 10 ** 6
	dt = (tf / nt) 
	kappa = 2 * np.pi * 10 * 10 ** 3
	K = 2 * np.pi * 325 * 10 ** 3
	for i in range(N):
		for j in range(N):
			print(ic(i, j, beta))
	#rho = evolve(N, tf, nt, beta, kappa, K)
