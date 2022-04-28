import numpy as np
import qutip
from matplotlib import pyplot as plt

if __name__ == "__main__":
	denmat = np.load("denmat.npy")
	N = denmat.shape[0]	
	rho = 0 * qutip.states.basis(N, 0) * qutip.states.basis(N, 0).dag() 
	for i in range(N):
		for j in range(N):
			rho += denmat[i,j] * qutip.states.basis(N, i) * qutip.states.basis(N, j).dag()
	x = np.linspace(-3, 3, N)	
	w = qutip.wigner(rho, x, x)	
	fig, ax = plt.subplots()
	cont = ax.contourf(x, x, w, 100)
	plt.show()

