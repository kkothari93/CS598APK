## main file
# this file glues all ops in different files and runs the solve

from ititree import *
from input_ import PlaneWave
from external import SqDomain
import numpy as np
import matplotlib.pyplot as plt


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

	u = {}
	solveBVP(a, in_.f, u)
	plot_solution(a,u, in_)
	b = SqDomain(a, 16, 224)
	us = b.solve(Tint, in_)

	print(us)



if __name__ == '__main__':
	main()