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

    def _init_(self, sw_c, ne_c, b = potfn, id_=0, levels=4, tic=1):
        self.root = Box(sw_c, ne_c, b)
        BoxList.append(self.root)
        self.id = id_
        self.__levels = levels
        if levels > 0:
            self.a, self.b = self._divide(tic)
        	self.isLeaf = False
        else:
        	self.isLeaf = True

    def _divide(self, tic):
        if tic:
            # vertical split
            alpha = BoxTree(sw_c, ((sw_c[0]+ne_c[0])/2.0, ne_c[1]),
                            id_=2*self.id_+1, levels=self.__levels - 1, tic=0)
            beta = BoxTree(((sw_c[0]+ne_c[0])/2, sw_c[1]), ne_c,
            				id_=2*self.id_+2, levels=self.__levels - 1, tic=0)

            self.j1a = np.concatenate((alpha.root.js, alpha.root.je, alpha.root.jw[1:]))
            self.j3a = np.append(alpha.root.jn, alpha.root.jw[0])

            self.j2b = np.concatenate((beta.root.je[1:], beta.root.jn, beta.root.jw))
            self.j3b = np.append(beta.root.js, beta.root.je[0])

        else:
        	# horizontal split
        	alpha = BoxTree(sw_c, (ne_c[0], (sw_c[1]+ne_c[1])/2.0),
                            id_=2*self.id_+1, levels=self.__levels - 1, tic=1)
            beta = BoxTree((sw_c[0], (sw_c[1]+ne_c[1])/2.0), ne_c
            				id_=2*self.id_+2, levels=self.__levels - 1, tic=1)

            self.j1a = np.concatenate((alpha.root.js, alpha.root.jn[1:], alpha.root.jw))
            self.j3a = np.append(alpha.root.je, alpha.root.jn[0])

            self.j2b = np.concatenate((beta.root.js[1:], beta.root.je, beta.root.jn))
            self.j3b = np.append(beta.root.jw, beta.root.js[0])

        return alpha, beta

    def merge(self):
    	if self.isLeaf:
    		return
    	else:
    		R11a = self.a[np.ix_(self.j1a, self.j1a)]
    		R13a = self.a[np.ix_(self.j1a, self.j3a)]
    		R31a = self.a[np.ix_(self.j3a, self.j1a)]
    		R33a = self.a[np.ix_(self.j3a, self.j3a)]

    		R22b = self.b[np.ix_(self.j2b, self.j2b)]
    		R23b = self.b[np.ix_(self.j2b, self.j3b)]
    		R32b = self.b[np.ix_(self.j3b, self.j2b)]
    		R33b = self.b[np.ix_(self.j3b, self.j3b)]

    		W = np.linalg.inv(np.eye(len(R33a)) - R33b @ R33a)
    		self.R = np.hstack((np.vstack((R11a + R13a @ W @ R33b @ R31b, 
    							-R23b @ (R31a + R33a @ W @ R33a @ R31a))),
    		 		 			np.vstack(-R13a @ W @ R32b, 
    							R22b @ (R23b @ R33a @ self.a.root.Q @ R32b))))
    		self.Sa = np.hstack((W @ R33b @ R31a, -W @ R32b))
    		self.Sb = -np.hstack((R33a + W @ R33b @ R31a, -W @ R32b))

    		del self.a.root.R, self.b.root.R
    		del R11a, R13a, R31a, R33a
    		del R22b, R23b, R32b, R33b

    	return



def build_sol(a):
	for box in a.BoxList[::-1]:
		if box.isLeaf:
			box.root._build_ops()

		else:
			box.merge()

	root = a.BoxList[0].root
	I = np.eye(len(root.R))
	Tint = -1j*root.k*np.linalg.inv(root.R - I) @ (root.R + I)


def solveBVP(a, f, u):
	for box in a.BoxList:
		if box.isLeaf:
			u[box.jt] = box.root.Y @ box.fT
		else:
			box.a.fT = box.Sa @ f[box.jt]
			box.b.fT = box.Sb @ f[box.jt]
	return 


