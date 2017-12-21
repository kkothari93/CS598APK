#
# ItI maps

__author__ = "Konik Kothari"

import numpy as np
import matplotlib.pyplot as plt
from iti import Box


def potfn(x):
    """Insert fav potential function over domain here"""
    # return np.exp(-160*np.linalg.norm(x, axis=0)**2)

    return np.zeros_like(x[0])


class BoxTree(object):
    """
    Tree object to solve interior variable media 
    Helmholtz Dirichlet problem
    """

    BoxList = []  # to get a list access for algo 1 later on

    def __init__(self, sw_c, ne_c, p=16, q=14,
                 b=potfn, id_=0, levels=2, tic=1):
        self.root = Box(sw_c, ne_c, b, p, q, isLeaf=levels == 0)
        self.BoxList.append(self)
        self.id_ = id_
        self.__levels = levels

        # initiate impedance to None
        # needed for solveBVP
        self.fT = None

        if levels > 0:
            self.a, self.b = self._divide(tic)
            self.isLeaf = False

        else:
            self.isLeaf = True
            self.pts = self.root.gauss_grid

    def __hash__(self):
        return self.id_

    def __str__(self):
        return str(self.id_)

    def reset(self):
        """short of implementing a boxlist class,
        (as that is not required here), this 
        function will erase the boxes generated
        yet and regenerate a list from the instance 
        it has been called. (i.e. the instance will 
        become the root)

        """
        self.BoxList = []

        from collections import deque

        q = deque([])
        node = self.root
        if node is not None:
            q.append(node)
            if not node.isLeaf:
                q.append(node.a)
                q.append(node.b)
            BoxList.append(q.popleft())
            try:
                node = q.popleft()
            except IndexError:
                node = None
        return self

    def _divide(self, tic):
        """Divides cells and builds the indices"""
        sw_c = self.root.sw_c
        ne_c = self.root.ne_c
        if tic:
            # vertical line split
            alpha = BoxTree(sw_c, ((sw_c[0]+ne_c[0])/2.0, ne_c[1]),
                            id_=2*self.id_+1, levels=self.__levels - 0.5, tic=0)
            beta = BoxTree(((sw_c[0]+ne_c[0])/2, sw_c[1]), ne_c,
                           id_=2*self.id_+2, levels=self.__levels - 0.5, tic=0)
        else:
            # horizontal line split
            alpha = BoxTree(sw_c, (ne_c[0], (sw_c[1]+ne_c[1])/2.0),
                            id_=2*self.id_+1, levels=self.__levels - 0.5, tic=1)
            beta = BoxTree((sw_c[0], (sw_c[1]+ne_c[1])/2.0), ne_c,
                           id_=2*self.id_+2, levels=self.__levels - 0.5, tic=1)

        return alpha, beta

    def __build_indices(self):
        """builds indices for merge process"""
        # aliasing for ease of use
        a = self.a.root
        b = self.b.root
        r = self.root

        lajsg, lajng, lajeg, lajwg = list(
            map(len, (a.jsg, a.jng, a.jeg, a.jwg)))
        lbjsg, lbjng, lbjeg, lbjwg = list(
            map(len, (b.jsg, b.jng, b.jeg, b.jwg)))

        assert lajsg+lajeg+lajng+lajwg == lbjsg+lbjeg+lbjng+lbjwg
        # figure out if a and b are on the right of each other
        # or are up and down
        updown = False
        if a.ne_c[1] < b.ne_c[1]:
            updown = True

        if updown:
            r.jsg = np.arange(lajsg)
            r.jeg = r.jsg[-1] + 1 + np.arange(lajeg+lbjeg)
            r.jng = r.jeg[-1] + 1 + np.arange(lbjng)
            r.jwg = r.jng[-1] + 1 + np.arange(lajwg+lbjwg)
        else:
            r.jsg = np.arange(lajsg + lbjsg)
            r.jeg = r.jsg[-1] + 1 + np.arange(lbjeg)
            r.jng = r.jeg[-1] + 1 + np.arange(lajng+lbjng)
            r.jwg = r.jng[-1] + 1 + np.arange(lajwg)

    def merge(self):
        """Merges ops from children"""
        if self.isLeaf:
            return
        else:
            alpha = self.a
            beta = self.b

            updown = False
            if alpha.root.ne_c[1] < beta.root.ne_c[1]:
                updown = True
            if not updown:
                self.j1a = np.concatenate(
                    (alpha.root.jsg, alpha.root.jng, alpha.root.jwg))
                self.j3a = alpha.root.jeg

                self.j2b = np.concatenate(
                    (beta.root.jsg, beta.root.jeg, beta.root.jng))
                self.j3b = beta.root.jwg

                self.pts = np.concatenate((
                    alpha.pts[:, alpha.root.jsg],
                    beta.pts[:, beta.root.jsg],
                    beta.pts[:, beta.root.jeg],
                    beta.pts[:, beta.root.jng],
                    alpha.pts[:, alpha.root.jng],
                    alpha.pts[:, alpha.root.jwg]), axis=1)

            else:
                self.j1a = np.concatenate(
                    (alpha.root.jsg, alpha.root.jeg, alpha.root.jwg))
                self.j3a = alpha.root.jng

                self.j2b = np.concatenate(
                    (beta.root.jeg, beta.root.jng, beta.root.jwg))
                self.j3b = beta.root.jsg

                self.pts = np.concatenate((
                    alpha.pts[:, alpha.root.jsg],
                    alpha.pts[:, alpha.root.jeg],
                    beta.pts[:, beta.root.jeg],
                    beta.pts[:, beta.root.jng],
                    beta.pts[:, beta.root.jwg],
                    alpha.pts[:, alpha.root.jwg]), axis=1)

            R11a = self.a.root.R[np.ix_(self.j1a, self.j1a)]
            R13a = self.a.root.R[np.ix_(self.j1a, self.j3a)]
            R31a = self.a.root.R[np.ix_(self.j3a, self.j1a)]
            R33a = self.a.root.R[np.ix_(self.j3a, self.j3a)]

            R22b = self.b.root.R[np.ix_(self.j2b, self.j2b)]
            R23b = self.b.root.R[np.ix_(self.j2b, self.j3b)]
            R32b = self.b.root.R[np.ix_(self.j3b, self.j2b)]
            R33b = self.b.root.R[np.ix_(self.j3b, self.j3b)]

            W = np.linalg.inv(np.eye(len(R33a)) - R33b @ R33a)

            WR33bR31a = W @ R33b @ R31a
            WR32b = W@R32b

            self.root.R = np.hstack((np.vstack((R11a + R13a @ WR33bR31a,
                                                -R23b @ (R31a + R33a @ W @ R33b @ R31a))),
                                     np.vstack((-R13a @ WR32b,
                                                R22b + R23b @ R33a @ WR32b))))

            self.Sa = np.concatenate((WR33bR31a, -WR32b), axis=1)
            self.Sb = -np.concatenate((R31a + R33a @ WR33bR31a, -WR32b), axis=1)

            del self.a.root.R, self.b.root.R
            del R11a, R13a, R31a, R33a
            del R22b, R23b, R32b, R33b

            self.__build_indices()

        return

    def sort(self):
        """Sorts boxes in boxlist by boxlist id
        required as the current division method works
        in dfs fashion
        """
        self.BoxList.sort(key=lambda x: x.id_, reverse=False)


def build_sol(a):
    """Builds solution operator(Tint) for internal domain"""
    a.sort()  # needed so that ops merge proceed in right fashion
    for box in a.BoxList[::-1]:
        if box.isLeaf:
            box.root.build_ops()

        else:
            box.merge()

    root = a.BoxList[0].root
    I = np.eye(len(root.R))
    Tint = -1j*root.k*np.linalg.inv(root.R - I) @ (root.R + I)
    
    return Tint


def solveBVP(a, f, u):
    """Solve internal variable coeff BVP"""
    for box in a.BoxList:
        if box.isLeaf:
            u[box.id_] = box.root.Y @ box.fT

        else:
            if box.id_ == 0:
                box.f1a = f(box.pts[:, box.j1a])
                box.f2b = f(box.pts[:, box.j2b])
                box.fT = np.concatenate((box.f1a, box.f2b), axis=0)
            else:
                box.f1a = box.fT[box.j1a]
                box.f2b = box.fT[box.j2b]

            f3a = box.Sa @ box.fT
            f3b = box.Sb @ box.fT

            box.a.fT = np.concatenate((box.f1a, f3a), axis=0)
            box.b.fT = np.concatenate((box.f2b, f3b), axis=0)
    return None


def plot_solution(a, u,in_):
    # TODO: BUGGY!!
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = fig.gca()

    for box in a.BoxList:
        if box.isLeaf:
            utemp = u[box.id_]
            pos = box.root.cheb_grid
            # utemp = in_.u_in(pos)
            # print(utemp.shape)
            ax.plot_trisurf(pos[0, :],
                         pos[1, :],
                         utemp.real,
                         cmap=cm.jet,
                         vmin=-1, vmax=1, shade=True)

    plt.show()


def test():
    """Tester will test build ops (Alg 1) of paper"""
    a = BoxTree((-0.5, -0.5), (0.5, 0.5))
    Tint = build_sol(a)
    print(Tint.shape)
    return a


if __name__ == '__main__':
    test()
