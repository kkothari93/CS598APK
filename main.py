## main file
# this file glues all ops in different files and runs the solve

from ititree import *
from input_ import PlaneWave
from external import SqDomain
import numpy as np
import matplotlib.pyplot as plt

# TODO: Index U and test external.py

def main():
	in_ = PlaneWave(40, np.array([1+1j*0]))
	a = BoxTree((-0.5, -0.5), (0.5, 0.5))
	Tint = build_sol(a)
	u = {}
	solveBVP(a, in_.f, u)
	b = SqDomain(a, 16, 224)
	us = b.solve(Tint, in_.u_in, in_.grad_u_in)



if __name__ == '__main__':
	main()



