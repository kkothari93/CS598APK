#
# External problem

__author__ = "Konik Kothari"

import scipy.special as fns
import scipy
import numpy as np
import matplotlib.pyplot as plt


class SqDomain(object):
    """
    Returns the parametrization of the square domain 
    given its southwest corner and northeast corner

    Input
    -----
    swc : complex value position of SW corner in the complex plane
    nec : complex value position of NE corner in the complex plane
    npanels : Number of panels to be used on the boundary
    nptsonbdry : Pass this parameter from the root node of interior solve

    """

    def __init__(self, swc, nec, npanels, nptsonbdry):
        self.swc = swc
        self.nec = nec
        self._p = npanels
        self._N = nptsonbdry
        self.n = np.ceil(self._N/self._p)
        self.curvquad()
        return self

    def gamma(self, s):
        q = np.pi/2
        side = np.real(self.nec-self.swc)
        conditions = [np.logical_and(s >= 0, s < q),
                      np.logical_and(s >= q, s < 2*q),
                      np.logical_and(s >= 2*q, s < 3*q),
                      s >= 3*q]
        return_vals = [s/q + self.swc,
                       side + side*1j*(s/q-1) + self.swc,
                       side*1j + side*(3-s/q) + self.swc,
                       side*1j*(4-s/q) + self.swc]
        a = np.where(conditions, return_vals, 0).sum(axis=0)
        return a

    def gammap(self, s):
        q = np.pi/2
        side = np.real(self.nec-self.swc)
        conditions = [np.logical_and(s >= 0, s < q),
                      np.logical_and(s >= q, s < 2*q),
                      np.logical_and(s >= 2*q, s < 3*q),
                      s >= 3*q]
        return_vals = [side*1/q,
                       side*1j/q,
                       side*-1/q,
                       side*-1j/q]
        a = np.where(conditions, return_vals, 0).sum(axis=0)
        return a

    def curvquad(self):
        # get Gauss points and weights
        pts, w = np.polynomial.legendre.leggauss(10)
        # fit to [0,1]
        pts = pts*0.5 + 0.5
        w /= 2
        se = 2*np.pi/self.n*np.arange(0, self.n+1)
        self.weights = np.tile(w*2*np.pi/self.n, (self.n, 1))
        s = np.zeros(self.n*self._p)
        for i in xrange(n):
            s[i*p:(i+1)*p] = se[i] + (se[i+1]-se[i])*pts

        self.s = s
        self.x = self.gamma(s)
        self.__sp = self.gammap(s)
        self.sp = abs(self.__sp)
        self.normals = -1j*self.__sp/self.sp
        return None

    def prep_operator(self,k,Tint):
        A = np.eye(self.__N)/2
        dv = self.x - self.x[:, None]
        d = np.abs(dv)
        self.costheta = np.real(np.conj(G.normals) @ dv)/d
        d *= k
        
        # $$ \dfrac{\partial H_n^{(1)}(z)}{\partial z} = 
        # \dfrac{n H_n^{(1)}(z)}{z} - H_{n+1}^{(1)}(z)$$
        
        # Kapur-Rokhlin correction

        # filling diagonal with 0s as per KR quad scheme
        D = -1j/4*fns.hankel1(1, d) @ self.costheta
        D = np.fill_diagonal(D, 0) @ self.sp @ self.weights
        
        S = 1j/4*fns.hankel1(0,d)
        S = np.fill_diagonal(S, 0) @ self.sp @ self.weights

        ## implementing a 6th order correction scheme here

        # taken from Alex Barnett's MPSPack lib
        g6 = [4.967362978287758, -16.20501504859126, 25.85153761832639,
              -22.22599466791883, 9.930104998037539, -1.817995878141594]

        corrections = np.ones((self.__N, self.__N))
        for i in range(self.__N):
            for j in range(1,7):
                corrections[i][i-j] = g[j-1]
                corrections[i][(i+j)%N] = g[j-1]
        S *= corrections
        D *= corrections

        A += S @ Tint - D
        
        self.S = S
        self.D = D
        self.A = A

        return A

    def solve(self, Tint, u_in, grad_u_in):
        rhs = self.S @ (grad_u_in(self.x) @ self.costheta - Tint@u_in(self.x))
        soln = scipy.sparse.linalg.gmres(self.A, rhs, tol=1e-7, restart=20)
        return soln

        


