#
# ItI maps

__author__ = "Konik Kothari"

import numpy as np
import matplotlib.pyplot as plt
from iti import Box


def potfn(x):
    return np.exp(-160*np.linalg.norm(x, axis=0)**2)


class BoxTree(object):

    BoxList = []  # to get a list access for algo 1 later on

    def __init__(self, sw_c, ne_c, b = potfn, id_=0, levels=4, tic=1):
        self.root = Box(sw_c, ne_c, b, isLeaf = levels==0)
        self.BoxList.append(self)
        self.id_ = id_
        self.__levels = levels
        if levels > 0:
            self.a, self.b = self._divide(tic)
            self.isLeaf = False
        else:
            self.isLeaf = True


    def reset(self):
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
        sw_c = self.root.sw_c
        ne_c = self.root.ne_c
        if tic:
            # vertical split
            alpha = BoxTree(sw_c, ((sw_c[0]+ne_c[0])/2.0, ne_c[1]),
                            id_=2*self.id_+1, levels=self.__levels - 1, tic=0)
            beta = BoxTree(((sw_c[0]+ne_c[0])/2, sw_c[1]), ne_c,
                            id_=2*self.id_+2, levels=self.__levels - 1, tic=0)

            self.j1a = np.concatenate((alpha.root.jsg, alpha.root.jeg, alpha.root.jwg))
            self.j3a = alpha.root.jng

            self.j2b = np.concatenate((beta.root.jeg, beta.root.jng, beta.root.jwg))
            self.j3b = beta.root.jsg

        else:
            # horizontal split
            alpha = BoxTree(sw_c, (ne_c[0], (sw_c[1]+ne_c[1])/2.0),
                            id_=2*self.id_+1, levels=self.__levels - 1, tic=1)
            beta = BoxTree((sw_c[0], (sw_c[1]+ne_c[1])/2.0), ne_c,
                            id_=2*self.id_+2, levels=self.__levels - 1, tic=1)

            self.j1a = np.concatenate((alpha.root.jsg, alpha.root.jng, alpha.root.jwg))
            self.j3a = alpha.root.jeg

            self.j2b = np.concatenate((beta.root.jsg, beta.root.jeg, beta.root.jng))
            self.j3b = beta.root.jwg

        return alpha, beta


    def merge(self):
        if self.isLeaf:
            return
        else:
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

        return

    def sort(self):
        self.BoxList.sort(key=lambda x: x.id_, reverse=False)

def build_sol(a):
    a.sort()
    for box in a.BoxList[::-1]:
        if box.isLeaf:
            box.root.build_ops()

        else:
            box.merge()

    root = a.BoxList[0].root
    I = np.eye(len(root.R))
    Tint = -1j*root.k*np.linalg.inv(root.R - I) @ (root.R + I)
    print(a.BoxList[0].Sa.shape)
    print(a.BoxList[0].Sb.shape)
    return Tint



def solveBVP(a, f, u):
    for box in a.BoxList:
        if box.isLeaf:
            u[box.jt] = box.root.Y @ box.fT
        else:
            box.a.fT = box.Sa @ f(box.cheb_grid)
            box.b.fT = box.Sb @ f(box.cheb_grid)
    return u


def test():
    a = BoxTree((-0.5, -0.5), (0.5, 0.5))
    Tint = build_sol(a)
    print(Tint.shape)
    return a


if __name__ == '__main__':
    test()
