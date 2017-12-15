# Input functions
# Boundary conditions
import numpy as np

class PlaneWave:
	"""Implements a plane wave class"""

    def __init__(self, k, w):
        self.k = k
        self.w = w

    def u_in(self, pts):
        return np.exp(1j*self.k*self.w @ pts)

    def grad_u_in(self, pts):
        return 1j*k*w * u_in(pts, self.k, self.w)

    def f(self, pts, eta):
        return self.u_in(pts) + 1j*eta*self.grad_u_in(pts)

    def g(self, pts, eta)
        return self.u_in(pts) + 1j*eta*self.grad_u_in(pts)
