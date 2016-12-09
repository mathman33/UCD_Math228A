from __future__ import division

from smoothers import GSRBBR
from operators import restriction, interpolation, compute_residual
from get_mesh import get_mesh
from copy import deepcopy
from time import clock
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import cPickle as pickle

def trivial_direct_solve(h,r):
    # h: grid spacing
    # r: residual input (1x1 matrix)

    # get the only item in the matrix
    R = r.item((0,0))
    # solve the equation
    sol = (-(h**2)/4)*R
    # return the solution, reshaped into a 1x1 matrix
    return np.asarray(sol).reshape((1,1))

def V_cycle(power,u,f,nu,X,Y,N,h,Ls):
    # power: the power of 2
    # u:     the grid to iterate
    # f:     the righthand side function to use
    # nu:    a 2-tuple of pre- and post-smooth numbers
    # X,Y:   the mesh
    # N:     number of points in the grid (including 0 and 1)
    # h:     grid spacing

    # copy the input to a new grid
    v = deepcopy(u)

    # presmooth nu_1 times
    for i in xrange(nu[0]):
        # smooth the grid
        v = GSRBBR(v,f,N,h)

    # Compute the residual
    r = compute_residual(v,f,Ls)

    # restrict the residual
    r = restriction(r,power)

    # if this is the smallest possible grid, direct solve
    if power == 2:
        # get the error analytically
        e = trivial_direct_solve(2*h,r)
    # otherwise, call the V_cycle code recursively
    else:
        # create a coarse mesh
        X_c,Y_c,N_c,h_c = get_mesh(power-1)
        # initialize a guess for the error
        e_guess = np.zeros((2**(power-1) - 1, 2**(power-1) - 1))
        # run the V_cycle code for the residual equation
        e = V_cycle(power-1, e_guess, r, nu, X_c, Y_c, N_c, h_c, Ls)

    # interpolate the error
    e = interpolation(e,power)

    # correct the solution
    v = v + e

    # postsmooth nu_2 times
    for i in xrange(nu[1]):
        # smooth the grid
        v = GSRBBR(v,f,N,h)

    # return the smoothed and iterated grid
    return v

