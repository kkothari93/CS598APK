#
# ItI maps

__author__ = "Konik Kothari"

import numpy as np
import matplotlib.pyplot as plt
from chebDiff import chebdif
import scipy.interpolate


class Box(object):
    """Implements the box object"""

    def __init__(self, sw_c, ne_c, b, p, q, isLeaf=False, k=40, id_=0):

        self.id = id_
        self.sw_c = sw_c if type(sw_c) is np.ndarray else np.array(sw_c)
        self.ne_c = ne_c if type(ne_c) is np.ndarray else np.array(ne_c)
        self.hx = -(ne_c[0]-sw_c[0])/2
        self.hy = -(ne_c[1]-sw_c[1])/2

        self.pot = b
        self.k = k
        self.isLeaf = isLeaf

        if isLeaf:
            self._p = p  # Size of cheb grid
            self._q = q  # Size of Gauss-grid

            self.cheb_grid = self._build_cheb_grid()
            self.gauss_grid, self.normals = self._build_gauss_edges()
        # self._plot_grid(self.cheb_grid)

        return

    def _ccw_ordering(self, pts, mp, x, y, q):
        """ Numbers cheb grid according to scheme in paper"""

        # south edge
        pts[0, :q-1] = mp[0] + x[:-1]
        pts[1, :q-1] = mp[1] + y[0]

        # east edge
        pts[0, q-1:2*(q-1)] = mp[0] + x[-1]
        pts[1, q-1:2*(q-1)] = mp[1] + y[:-1]

        # north edge
        pts[0, 2*(q-1):3*(q-1)] = mp[0] + x[::-1][:-1]
        pts[1, 2*(q-1):3*(q-1)] = mp[1] + y[-1]

        # west edge
        pts[0, 3*(q-1):4*(q-1)] = mp[0] + x[0]
        pts[1, 3*(q-1):4*(q-1)] = mp[1] + y[::-1][:-1]

        return None

    def _build_cheb_grid(self):
        """ Returns a p x p Chebyshev grid """
        p = self._p
        j = np.arange(p) + 1
        xx = self.hx*np.cos(np.pi*(j-1)/(p-1))
        yy = self.hy*np.cos(np.pi*(j-1)/(p-1))
        mp = (self.sw_c+self.ne_c)/2
        pts = np.zeros((2, p*p))

        self._ccw_ordering(pts, mp, xx, yy, p)
        self.js = np.arange(p-1)
        self.je = np.arange(p-1) + p-1
        self.jn = np.arange(p-1) + 2*(p-1)
        self.jw = np.arange(p-1) + 3*(p-1)

        self.jb = np.concatenate((self.js, self.je, self.jn, self.jw))

        # interior points
        for i in range(p-2):
            for j in range(p-2):

                pts[0, 4*(p-1)+i*(p-2)+j] = mp[0] + xx[j+1]
                pts[1, 4*(p-1)+i*(p-2)+j] = mp[1] + yy[i+1]

        self.ji = np.arange(4*(p-1), p*p)

        return pts

    def _build_gauss_edges(self):
        """ Builds the edge gauss grid """
        q = self._q
        ns = np.array([0,-1])
        ne = np.array([1, 0])
        nn = np.array([0, 1])
        nw = np.array([-1, 0])

        x, _ = np.polynomial.legendre.leggauss(q)

        # scale to our case
        # leggauss gives points on interval [-1,1]
        # (i.e. of length 2). Our box has side length 2*h
        # and midpoint non-zero.
        xx = x/2*2*abs(self.hx)
        yy = x/2*2*abs(self.hy)
        pts = np.zeros((2, 4*q))
        normals = np.zeros((2, 4*q))
        
        mp = (self.sw_c+self.ne_c)/2

        # south edge
        pts[0, :q] = mp[0] + xx
        pts[1, :q] = self.sw_c[1]
        normals[:,:q] = ns[:, None]
        self.jsg = np.arange(q)

        # east edge
        pts[0, q:2*q] = self.ne_c[0]
        pts[1, q:2*q] = mp[1] + yy
        normals[:, q:2*q] = ne[:, None]
        self.jeg = np.arange(q, 2*q)

        # north edge
        pts[0, 2*q:3*q] = mp[0] + xx[::-1]
        pts[1, 2*q:3*q] = self.ne_c[1]
        normals[:, 2*q:3*q] = nn[:, None]
        self.jng = np.arange(2*q, 3*q)

        # west edge
        pts[0, 3*q:4*q] = self.sw_c[0]
        pts[1, 3*q:4*q] = mp[1] + yy[::-1]
        normals[:, 3*q:4*q] = nw[:, None]
        self.jwg = np.arange(3*q, 4*q)

        return pts, normals

    def _permute(self, A):
        """
        Takes a p**2 x p**2 matrix and permutes it to have structure:

        A = | A_bb | A_bi |
            | ----------- |
            | A_ib | A_ii |

        """

        p = int(np.sqrt(len(A)))
        assert p == self._p

        # get permuted id's
        ids = self.__permute_ids()

        return A[np.ix_(ids, ids)]

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

    def __permute_ids(self):
        p = self._p
        ids = np.arange(p, dtype=int) #0,1,...,p-1
        j = np.arange(1,p, dtype= int)+1 
        ids = np.concatenate((ids, j*p-1)) #2p-1,3p-1,...,p**2-1
        ids = np.concatenate((ids, p*p-j)) #p**2-2,p**2-3,...,p**2-p
        ids = np.concatenate((ids, np.arange(p-2,0,-1, dtype=int)*p))

        ids = np.concatenate((ids, np.zeros((p-2)**2, dtype=int)))

        for i in range(p-2):
            for j in range(p-2):
                ids[4*(p-1) + i*(p-2) + j] = (i+1)*p + j+1

        return ids


    def _cons_in_x(self):
        return 3*(self.cheb_grid[1,:])**2 

    def _cons_in_y(self):
        return 3*(self.cheb_grid[0,:])**2

    def _harmonic(self):
        return np.log(self._cons_in_x() + self._cons_in_y())

    def build_ops(self):
        """Generates ops for the box"""
        p = self._p
        # chebdif tested with _cons_in_x/y functions
        # works well
        D = chebdif(p, 1)
        D = D.reshape((p, p))


        # D on chebyshev points
        # https://www.nada.kth.se/~olofr/Approx/BarycentricLagrange.pdf
        # !! DOES NOT WORK: DON'T KNOW WHY !! :(
        # delj = np.ones(p)
        # delj[0] = delj[-1] = 0.5
        # j = np.arange(p)+1
        # x = np.cos(np.pi*(j-1)/(p-1))[::-1]
        # w = (-1)**j * delj
        # dij = x - x[:, None]
        # np.fill_diagonal(dij, 1.0)
        # D = np.outer(1/w, w) * 1/dij
        # # print(np.diag(D))

        Dx = self._permute(np.kron(np.eye(p), D))/self.hx
        Dy = self._permute(np.kron(D, np.eye(p)))/self.hy
        DD = self._permute(np.diag(self.k**2 * (1 - self.pot(self.cheb_grid))))

        # wave operator
        A = Dx @ Dx + Dy @ Dy + DD
        # print((Dx @ Dx + Dy @ Dy) @ self._harmonic())


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

        self.P = np.kron(np.eye(4), P[:-1, :])

        # Cheb to Gauss mapping
        self.Q = self.interpolation(self.gauss_grid[0][:self._q],
                                    self.cheb_grid[0][:self._p])

        self.Y = X @ self.P

        
        # plt.imshow(self.Y.real, vmin=self.Y.real.min(), vmax=self.Y.real.max(), interpolation='nearest')
        # plt.colorbar()
        # plt.show()


        # gauss will use both end points
        jsp = np.append(self.js, self.je[0])
        jep = np.append(self.je, self.jn[0])
        jnp = np.append(self.jn, self.jw[0])
        jwp = np.append(self.jw, self.js[0])
        jbp = np.concatenate((jsp, jep, jnp, jwp))
        G = np.vstack((-Dy[jsp, :], Dx[jep, :], Dy[jnp, :],
                       -Dx[jwp, :])) - 1j*self.k*np.eye(p*p)[jbp, :]

        self.R = np.kron(np.eye(4), self.Q) @ G @ self.Y


        return A


    def _plot_grid(self, grid):
        """Plot a grid of points with index label"""
        fig, ax = plt.subplots()
        ax.scatter(grid[0, :], grid[1, :])

        for i, pt in enumerate(grid.T):
            ax.annotate(str(i), (pt[0], pt[1]))

        plt.show()
        return


def test():
    """Tests the box class"""
    from ititree import potfn
    from input_ import PlaneWave
    in_ = PlaneWave(40, np.array([1+0j]))
    a = Box((0.5, 0.5), (0.625, 0.625), potfn, p=16, q=14, isLeaf=True)
    A = a.build_ops()
    q = 14
    R = a.R
    pts, normals = a.gauss_grid, a.normals
    print(np.real(in_.grad_u_in(pts)*np.conj(normals)))
    # print(np.linalg.eigvals(R))
    lhs = R @ in_.f(pts, normals)
    rhs = in_.g(pts, normals)
    plt.scatter(lhs.real, lhs.imag, color='r')
    plt.scatter(rhs.real, rhs.imag, color='b')
    plt.show()
    print(np.linalg.norm(lhs-rhs))
    # print(A @ in_.u_in(a.cheb_grid))


if __name__ == "__main__":
    a = test()

# def test_interp():
#     """Lagrange interpolation tester"""
#     a = 
#     j = np.arange(16) + 1
#     xt = ((np.cos(np.pi*(j-1)/8)[::-1]) + 1)/2.0
#     xs, _ = np.polynomial.legendre.leggauss(14)
#     xs = (xs+1)/2
#     ans = interpolation(xt, xs)
#     y = xs**2
#     yp = ans @ y
#     plt.plot(xs, y)
#     plt.plot(xt, yp, 'r')
#     plt.show()


 
# if __name__ == "__main__":
    # test_interp()