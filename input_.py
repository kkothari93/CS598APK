# Input functions
# Boundary conditions
import numpy as np


class PlaneWave:
    """Implements a plane wave class"""

    def __init__(self, k, w):
        self.k = k
        self.w = w

    def __complexify(self, pts):
        return pts[0,:] + 1j*pts[1,:]

    def u_in(self, pts):
        if pts.shape[0]==2:
            pts = self.__complexify(pts)
        return np.exp(1j*self.k*np.real(np.conj(self.w)*pts))

    def grad_u_in(self, pts):
        if pts.shape[0]==2:
            pts = self.__complexify(pts)
        return 1j*self.k*self.w *self.u_in(pts)

    def f(self, pts):
        if pts.shape[0]==2:
            pts = self.__complexify(pts)
        return self.u_in(pts) + 1j*self.k*self.grad_u_in(pts)

    def g(self, pts):
        if pts.shape[0]==2:
            pts = self.__complexify(pts)
        return self.u_in(pts) - 1j*self.k*self.grad_u_in(pts)
