# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 10:17:35 2017

@author: konik
"""

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.linalg import eig
from scipy.linalg import toeplitz

def chebdif(N,M):
    '''    
    Calculate differentiation matrices using Chebyshev collocation.
      
    Returns the differentiation matrices D1, D2, .. DM corresponding to the 
    M-th derivative of the function f, at the N Chebyshev nodes in the 
    interval [-1,1].   
    
    Parameters
    ----------
     
    N   : int 
          number of grid points
         
    M   : int
          maximum order of the derivative, 0 < M <= N - 1
    Returns
    -------
    x  : ndarray
         N x 1 array of Chebyshev points 
         
    DM : ndarray
         M x N x N  array of differentiation matrices 
        
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the 
    1st, 2nd, ... M-th derivates on a Chebyshev grid of N points. The 
    matrices are constructed by differentiating N-th order Chebyshev 
    interpolants.  
    
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    
    .. math::
    
    f^{(m)}_i = D^{(m)}_{ij}f_j
     
    The code implements two strategies for enhanced accuracy suggested by 
    W. Don and S. Solomonoff :
    
    (a) the use of trigonometric  identities to avoid the computation of
    differences x(k)-x(j) 
    
    (b) the use of the "flipping trick"  which is necessary since sin t can 
    be computed to high relative precision when t is small whereas sin (pi-t) 
    cannot.
    
    It may, in fact, be slightly better not to implement the strategies 
    (a) and (b). Please consult [3] for details.
    
    This function is based on code by Nikola Mirkov 
    http://code.google.com/p/another-chebpy
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
 
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix 
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487 
           
    Examples
    --------
    
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as 
    
    >>> N = 32; M = 2; pi = np.pi
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.chebdif(N, M)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2*pi*x)              # function at Chebyshev nodes
    >>> plot(x, y, 'r', x, D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper left')
    '''

    if M >= N:
        raise Exception('numer of nodes must be greater than M')
        
    if M <= 0:
         raise Exception('derivative order must be at least 1')

    DM = np.zeros((M,N,N))
    
    n1 = N//2; n2 = int(np.ceil(N/2.))     # indices used for flipping trick
    k = np.arange(N)                        # compute theta vector
    th = k*np.pi/(N-1)

    # Compute the Chebyshev points

    #x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way   
    x = np.sin(np.pi*((N-1)-2*np.linspace(N-1,0,N))/(2*(N-1)))   # W&R way
#    x = x[::-1]
    
    # Assemble the differentiation matrices
    T = np.tile(th/2,(N,1))
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)               # trigonometric identity
    DX[n1:,:] = -np.flipud(np.fliplr(DX[0:n2,:]))    # flipping trick
    DX[range(N),range(N)]=1.                         # diagonals of D
    DX=DX.T

    C = toeplitz((-1.)**k)           # matrix with entries c(k)/c(j)
    C[0,:]  *= 2
    C[-1,:] *= 2
    C[:,0] *= 0.5
    C[:,-1] *= 0.5

    Z = 1./DX                        # Z contains entries 1/(x(k)-x(j))
    Z[range(N),range(N)] = 0.        # with zeros on the diagonal.          

    D = np.eye(N)                    # D contains differentiation matrices.
                                          
    for ell in range(M):
        D = (ell+1)*Z*(C*np.tile(np.diag(D),(N,1)).T - D)      # off-diagonals    
        D[range(N),range(N)]= -np.sum(D,axis=1)        # negative sum trick
        DM[ell,:,:] = D                                # store current D in DM

    return DM

if __name__ == '__main__':
    p = chebdif(5, 1)