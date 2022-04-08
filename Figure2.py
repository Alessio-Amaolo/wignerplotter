import numpy as np
import matplotlib.pyplot as plt 
from numba import njit 

def ground_q0(alpha):
	return (1/np.pi)*np.exp(-np.abs(alpha) ** 2)

def ground_q1(alpha):
	return (1/np.pi)*np.exp(-np.abs(alpha) ** 2) * np.abs(alpha) ** 2

def plot_data(grid, fun):
	x, y = grid 
	nx, ny = y.shape
	p = x * 0
	for i in range(nx):
		for j in range(ny):
			z = complex(x[i,j], y[i,j])
			p[i, j] = fun(z)
	return p

if __name__ == '__main__':
	functions = [ground_q0, ground_q1]
	titles = [r"Ground State $Q_0(\alpha)$", r"Ground State $Q_1(\alpha)$"]
	filenames = ["Fig2b", "Fig2d"]
	for i in range(2):
		fun = functions[i]
		title = titles[i]
		filename = filenames[i]
		grid = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101))
		data = plot_data(grid, fun)
		plt.figure(figsize=(6,6))
		plt.xlabel(r"Re[$\alpha$]")
		plt.ylabel(r"Im[$\alpha$]")
		#plt.contour(grid[0], grid[1], data[:,:,i])
		plt.pcolormesh(data) #, vmin=-0.1/np.pi, vmax=1/np.pi)
		plt.title(title)
		plt.savefig(filename)
		plt.clf()
		plt.close()
	titles = [r"Ground State $Q_0($Re$[\alpha])$", r"Ground State $Q_1($Re$[\alpha])$"]
	filenames = ["Fig2c", "Fig2e"]
	ylabels = [r"$Q_0($Re$[\alpha])$", r"$Q_1($Re$[\alpha])$"]
	for i in range(2):
		fun = functions[i]
		title = titles[i]
		filename = filenames[i]
		ylabel = ylabels[i]
		xs = np.linspace(-2, 2, 101)
		fig, ax = plt.subplots()
		ax.plot(xs, fun(xs))
		ax.set_xlabel(r"Re[$\alpha$]")
		ax.set_ylabel(ylabel)
		ax.set_title(title)
		fig.savefig(filename)
	



    
