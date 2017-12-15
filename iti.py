#
# ItI maps

__author__ = "Konik Kothari"

import numpy as np
import matplotlib.pyplot as plt
from chebDiff import chebdif
import scipy.interpolate


class Box(object):

    def __init__(self, sw_c, ne_c, b, isLeaf=False, k=40, id_=0):
        self.id = id_
        self.sw_c = sw_c if type(sw_c) is np.ndarray else np.array(sw_c)
        self.ne_c = ne_c if type(ne_c) is np.ndarray else np.array(ne_c)
        self.h = -(ne_c[0]-sw_c[0])/2

        self.pot = b
        self.k = k
        self.isLeaf = isLeaf

        if 1:
            self._p = 16  # Size of cheb grid
            self._q = 14  # Size of Gauss-grid

            self.cheb_grid = self._build_cheb_grid()
            self.gauss_grid = self._build_gauss_edges()
            # self._plot_grid(self.gauss_grid)

        return

    def _ccw_ordering(self, pts, mp, x, q):
        ''' Common ccw ordering of boundary points abstracted as a separate
        function in order to maximize reusability

        '''

        # south edge
        pts[0, :q-1] = mp[0] + x[:-1]
        pts[1, :q-1] = mp[1] + x[0]

        # east edge
        pts[0, q-1:2*(q-1)] = mp[0] + x[-1]
        pts[1, q-1:2*(q-1)] = mp[1] + x[:-1]

        # north edge
        pts[0, 2*(q-1):3*(q-1)] = mp[0] + x[::-1][:-1]
        pts[1, 2*(q-1):3*(q-1)] = mp[1] + x[-1]

        # west edge
        pts[0, 3*(q-1):4*(q-1)] = mp[0] + x[0]
        pts[1, 3*(q-1):4*(q-1)] = mp[1] + x[::-1][:-1]

        return None

    def _build_cheb_grid(self):
        """ Returns a p x p Chebyshev grid
        Input
        -----
        p : int
        Number of Chebyshev points (Default=16)
        """
        p = self._p
        j = np.arange(p) + 1
        xj = self.h*np.cos(np.pi*(j-1)/(p-1))
        mp = (self.sw_c+self.ne_c)/2
        pts = np.zeros((2, p*p))

        self._ccw_ordering(pts, mp, xj, p)
        self.js = np.arange(p-1)
        self.je = np.arange(p-1) + p-1
        self.jn = np.arange(p-1) + 2*(p-1)
        self.jw = np.arange(p-1) + 3*(p-1)

        self.jb = np.concatenate((self.js, self.je, self.jn, self.jw))

        # interior points
        for i in range(p-2):
            for j in range(p-2):

                pts[0, 4*(p-1)+i*(p-2)+j] = mp[0] + xj[j+1]
                pts[1, 4*(p-1)+i*(p-2)+j] = mp[1] + xj[i+1]

        self.ji = np.arange(4*(p-1), p*p)

        return pts

    def _build_gauss_edges(self):
        q = self._q
        x, _ = np.polynomial.legendre.leggauss(q)

        # scale to our case
        # leggauss gives points on interval [-1,1]
        # (i.e. of length 2). Our box has side length 2*h
        # and midpoint non-zero.
        x = x/2*2*abs(self.h)
        pts = np.zeros((2, 4*q))
        mp = (self.sw_c+self.ne_c)/2

        # south edge
        pts[0, :q] = mp[0] + x
        pts[1, :q] = self.sw_c[1]
        self.jsg = np.arange(q)

        # east edge
        pts[0, q:2*q] = self.ne_c[0]
        pts[1, q:2*q] = mp[1] + x
        self.jeg = np.arange(q,2*q)

        # north edge
        pts[0, 2*q:3*q] = mp[0] + x[::-1]
        pts[1, 2*q:3*q] = self.ne_c[1]
        self.jng = np.arange(2*q,3*q)
        
        # west edge
        pts[0, 3*q:4*q] = self.sw_c[0]
        pts[1, 3*q:4*q] = mp[1] + x[::-1]
        self.jwg = np.arange(3*q,4*q)

        return pts

    def _permute(self, A):
        """
        Takes a p**2 x p**2 matrix and permutes it to have structure:

        A = | A_bb | A_bi |
            | ----------- |
            | A_ib | A_ii |

        """
        p = int(np.sqrt(len(A)))
        assert p == self._p

        # first find the corner indices
        js = np.arange(p-1)
        je = np.arange(1, p)*p - 1
        jn = (p-1)*p + np.arange(p-1, 0, -1)
        jw = np.arange(p-1, 0, -1)*p

        jb = np.concatenate((js, je, jn, jw))
        # print(jb)

        # now find interior indices
        ji = [i*p+j for i in range(1, p-1) for j in range(1, p-1)]
        ji = np.array(ji)

        temp = np.zeros_like(A)
        # row permute
        temp[:len(jb), :len(jb)] = A[np.ix_(jb, jb)]
        temp[:len(jb), len(jb):] = A[np.ix_(jb, ji)]
        temp[len(jb):, :len(jb)] = A[np.ix_(ji, jb)]
        temp[len(jb):, len(jb):] = A[np.ix_(ji, ji)]

        A = temp

        del temp

        return A

    def interpolation(self, xt, xs, eps=1e-10):
        """Constructs the interpolation matrix 
        from xt (target) to xs (source) points

        Returns a numpy.ndarray of shape 
            len(xt) $$\times$$  len(xs)
        """
        p, q = map(len, [xt, xs])

        # lp --> Basis function derivative
        g = xs[:, None] - xs
        np.fill_diagonal(g, 1)
        lp = np.prod(g, axis=1)
        # w --> weights per source pt
        w = 1.0/lp

        # dts --> distance from target to source
        dts = xt[:, None]-xs

        # lagrange basis polynomial
        l = np.prod(dts, axis=1)

        # lagrange interpolation matrix
        # reshape required for numpy broadcast
        L = l.reshape(-1, 1) * (1/dts) * w

        # if xt and xs are very close
        # no need to interpolate
        problempts = np.abs(dts) < eps
        L[problempts] = 1

        return L

    def build_ops(self):
        p = self._p
        D = chebdif(p, 1)
        D = D.reshape((p, p))/abs(self.h)
        Dx = self._permute(np.kron(np.eye(p), D))
        Dy = self._permute(np.kron(D, np.eye(p)))
        DD = self._permute(np.diag(self.k**2 * (1 - self.pot(self.cheb_grid))))

        # wave operator
        A = Dx @ Dx + Dy @ Dy + DD

        # normal derivative
        N = np.vstack((-Dy[self.js, :], Dx[self.je, :],
                       Dy[self.jn, :], -Dx[self.jw, :]))

        # Outgoing impedance operator
        F = N + 1j*self.k * np.eye(p*p)[self.jb, :]

        # linear system
        B = np.vstack((F, A[self.ji, :]))

        # Solution matrix
        X = np.linalg.inv(B) @ np.vstack((
            np.eye(4*p-4),
            np.zeros(((p-2)**2, 4*p-4)))
        )

        # Gauss to Cheb mapping
        P = self.interpolation(self.cheb_grid[0][:self._p],
                               self.gauss_grid[0][:self._q])
        
        self.P = np.kron(np.eye(4),P[:-1, :])

        # Cheb to Gauss mapping
        self.Q = self.interpolation(self.gauss_grid[0][:self._q],
                               self.cheb_grid[0][:self._p])

        Y = X @ self.P

        # gauss will use both end points
        jsp = np.append(self.js, self.je[0])
        jep = np.append(self.je, self.jn[0])
        jnp = np.append(self.jn, self.jw[0])
        jwp = np.append(self.jw, self.js[0])
        jbp = np.concatenate((jsp, jep, jnp, jwp))
        G = np.vstack((-Dy[jsp, :], Dx[jep, :], Dy[jnp, :], -
                       Dx[jwp, :])) - 1j*self.k*np.eye(p*p)[jbp, :]

        self.R = np.kron(np.eye(4), self.Q) @ G @ Y

        return

    def _plot_grid(self, grid):
        """Plot a grid of points"""
        fig, ax = plt.subplots()
        ax.scatter(grid[0, :], grid[1, :])

        for i, pt in enumerate(grid.T):
            ax.annotate(str(i), (pt[0], pt[1]))

        plt.show()
        return


def test():
    from ititree import potfn
    a = Box((-0.5, -0.5), (0.5, 0.5), potfn, isLeaf=True)
    return a


def test_interp():
    j = np.arange(16) + 1
    xt = ((np.cos(np.pi*(j-1)/8)[::-1]) + 1)/2.0
    xs, _ = np.polynomial.legendre.leggauss(14)
    xs = (xs+1)/2
    ans = interpolation(xt, xs)
    y = xs**2
    yp = ans @ y
    plt.plot(xs, y)
    plt.plot(xt, yp, 'r')
    plt.show()


if __name__ == "__main__":
    # a = test()
    test_interp()
