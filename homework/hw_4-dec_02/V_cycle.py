from __future__ import division

from smoothers import GSRB
from operators import restriction, interpolation
from get_mesh import get_mesh
from copy import deepcopy
import scipy.sparse
import scipy.sparse.linalg
import numpy as np


def disc_lapl(N,h):
    off_diag = 1/(h**2)*np.ones(N**2)
    diag = -4/(h**2)*np.ones(N**2)
    A = np.vstack((off_diag,off_diag,diag,off_diag,off_diag))
    A = scipy.sparse.dia_matrix((A,[-N,-1,0,1,N]),shape=(N**2,N**2))
    A = scipy.sparse.csr_matrix(A)
    return A

def trivial_direct_solve(h,r):
    return np.asarray((-(h**2)/4)*r.item((0,0))).reshape((1,1))

def V_cycle(power,u,f,nu):
    v = deepcopy(u)
    X,Y,N,h = get_mesh(power)

    # Smooth
    for i in xrange(nu[0]):
        v = GSRB(v,f,N,h)

    # Compute Residual
    L = disc_lapl(N-2,h)
    r = (f.flatten() - L.dot(v.flatten())).reshape((int(2**power - 1),int(2**power - 1)))

    # Restrict
    r = restriction(r)

    # Solve
    if power == 2:
        e = trivial_direct_solve(2*h,r)
    else:
        e_guess = np.zeros((2**(power-1) - 1, 2**(power-1) - 1))
        e = V_cycle(power-1,e_guess,r, nu)

    # Interpolate the error
    e = interpolation(e,power)

    # Correct the solution
    v = v + e

    # Smooth
    for i in xrange(nu[1]):
        v = GSRB(v,f,N,h)

    return v

