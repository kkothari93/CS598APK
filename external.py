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

    def __init__(self, btree, npanels, nptsonbdry):
        self.swc = btree.root.sw_c[0]+1j*btree.root.sw_c[1]
        self.nec = btree.root.ne_c[0]+1j*btree.root.ne_c[1]
        self._p = npanels
        self.__prepped = False
        self._N = nptsonbdry
        self.n = int(np.ceil(self._N/self._p))
        self.curvquad()

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
        return_vals = [side*np.ones_like(s)/q,
                       side*np.ones_like(s)*1j/q,
                       side*np.ones_like(s)*-1/q,
                       side*np.ones_like(s)*-1j/q]

        a = np.where(conditions, return_vals, 0).sum(axis=0)
        return a

    def curvquad(self):
        # get Gauss points and weights
        # fit to [0,1]
        p = self._p
        n = self.n
        pts, w = np.polynomial.legendre.leggauss(p)
        pts = pts*0.5 + 0.5
        w /= 2
        se = 2*np.pi/n*np.arange(0, n+1)
        self.weights = np.tile(w*2*np.pi/n, n)

        s = np.zeros(n*p)
        for i in range(n):
            s[i*p:(i+1)*p] = se[i] + (se[i+1]-se[i])*pts

        self.s = s
        self.x = self.gamma(s)
        self.__sp = self.gammap(s)
        self.sp = np.abs(self.__sp)
        self.normals = -1j*self.__sp/self.sp
        return None

    def prep_operator(self, k, Tint):
        A = np.eye(self._N, dtype='complex128')
        dv = self.x - self.x[:, None]
        d = np.abs(dv)
        np.fill_diagonal(d, 1)
        self.costheta = np.real(np.conj(self.normals) * dv)/d
        d *= k

        # $$ \dfrac{\partial H_n^{(1)}(z)}{\partial z} =
        # \dfrac{n H_n^{(1)}(z)}{z} - H_{n+1}^{(1)}(z)$$

        # Kapur-Rokhlin correction

        # filling diagonal with 0s as per KR quad scheme
        D = -1j/4*fns.hankel1(1, d) @ self.costheta
        np.fill_diagonal(D, 0)
        # print("weights shape: %s"%str(self.weights.shape))
        D = D * self.sp * self.weights

        S = 1j/4*fns.hankel1(0, d)
        np.fill_diagonal(S, 0)
        S = S * self.sp * self.weights
        # print(S)

        # implementing a 6th order correction scheme here

        # taken from Alex Barnett's MPSPack lib
        g6 = [4.967362978287758, -16.20501504859126, 25.85153761832639,
              -22.22599466791883, 9.930104998037539, -1.817995878141594]

        corrections = np.ones((self._N, self._N))
        for i in range(self._N):
            for j in range(1, 7):
                corrections[i][i-j] = g6[j-1]
                corrections[i][(i+j) % self._N] = g6[j-1]

        S *= 2*corrections
        D *= 2*corrections

        A += S @ Tint.astype('complex128') - D

        self.S = S
        self.D = D
        self.A = A

        self.Text = np.linalg.inv(S) @ (D-np.eye(len(D))/2)

        eigvals = np.linalg.eigvals(A)
        plt.scatter(eigvals.real, eigvals.imag)
        plt.xlabel("Re($\lambda$)")
        plt.xlabel("Im($\lambda$)")
        plt.show()

        return A

    def solve(self, Tint, domain):
        k, u_in, grad_u_in = domain.k, domain.u_in, domain.grad_u_in
        if not self.__prepped:
            self.prep_operator(k, Tint)
        self.ui = u_in(self.x)
        self.uin = grad_u_in(self.x)
        rhs = self.S @ (self.uin @ self.costheta - Tint@ self.ui)
        # print(rhs)
        soln = scipy.sparse.linalg.gmres(self.A, rhs, tol=1e-7, restart=20)
        self.us = soln
        self.usn = Tint @ (self.ui + self.us) - self.uin

        return soln
