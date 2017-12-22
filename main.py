## main file
# this file glues all ops in different files and runs the solve

from ititree import *
from input_ import PlaneWave
from external import SqDomain
import numpy as np
import matplotlib.pyplot as plt

# TODO: Index U and test external.py

# def _plot_grid(grid):
#     """Plot a grid of points with index label"""
#     fig, ax = plt.subplots()
#     ax.scatter(grid[0, :], grid[1, :])

#     for i, pt in enumerate(grid.T):
#     	if i%4==0: ax.annotate(str(i), (pt[0], pt[1]))

#     plt.show()
#     return

def main():
	in_ = PlaneWave(40, np.array([1+0j]))
	a = BoxTree((1.0, 1.0), (1.25, 1.25))
	Tint, R = build_sol(a)

	plt.imshow(R.real, vmin=R.real.min(), vmax=R.real.max(), interpolation='nearest')
	plt.colorbar()
	plt.show()
	q = 56
	normals = np.zeros((2,224))
	normals[:,:q] = np.array([[0],[-1]]) 
	normals[:,q:2*q] = np.array([[1],[0]]) 
	normals[:,2*q:3*q] = np.array([[0],[1]]) 
	normals[:,3*q:4*q] = np.array([[-1],[0]]) 
	pts = a.BoxList[0].pts
	print(np.linalg.eigvals(R))
	lhs = R @ in_.f(pts, normals)
	rhs = in_.g(pts, normals)
	plt.scatter(lhs.real, lhs.imag, color='r')
	plt.scatter(rhs.real, rhs.imag, color='b')
	plt.show()
	print(np.linalg.norm(lhs-rhs))

	u = {}
	solveBVP(a, in_.f, u)
	plot_solution(a,u, in_)
	# b = SqDomain(a, 16, 224)
	# us = b.solve(Tint, in_)

	# print(us)



if __name__ == '__main__':
	main()