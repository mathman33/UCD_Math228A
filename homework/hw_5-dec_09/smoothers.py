from __future__ import division

import numpy as np
from copy import deepcopy

def get_checkerboard_of_size(s):
    # s: size of grid

    # make a row of 1 0 1 0 ..
    red = np.r_[int(s/2)*[1,0] + [1]]
    # make a row of 0 1 0 1 ..
    black = np.r_[int(s/2)*[0,1] + [0]]
    # put the rows in one list
    t = int(s/2)*(red, black)
    t += red,
    # stack the rows in a matrix
    checkerboard = np.row_stack(t)
    # return the checkerboard
    return checkerboard

def GS_iteration(inds, u, rhs, N, h):
    # inds: which indices to iteration
    # u:    the grid to smooth
    # rhs:  which right hand side function to use
    # N:    the size of the grid
    # h:    the grid spacing

    # copy the grid to save data from un-iterated indices
    # v = deepcopy(u)
    # loop over relevant indices
    for (i,j) in inds:
        # get the data from the grid point above (0 if top row)
        first  = 0 if i == 0 else u[i-1][j]
        # get the data from the grid point to the left (0 if first column)
        second = 0 if j == 0 else u[i][j-1]
        # get the data from the grid point below (0 if bottom row)
        third  = 0 if i == N else u[i+1][j]
        # get the data from the grid point to the right (0 if last column)
        fourth = 0 if j == N else u[i][j+1]
        # replace data point
        u[i][j] = (1/4)*(first + second + third + fourth - (h**2)*rhs[i][j])
    # return replaced grid
    return u

def GSRB(u,rhs,N,h):
    # u:   the grid to smooth
    # rhs: the right hand side function to use
    # N:   the size of the grid
    # h:   the grid spacing

    # Get a checkerboard of 1s and 0s
    checkerboard = get_checkerboard_of_size(u.shape[0])
    # Gather the indices of the "red" and "black" points
    red_indices = np.where(checkerboard == 1)
    black_indices = np.where(checkerboard == 0)
    red_indices = zip(red_indices[0],red_indices[1])
    black_indices = zip(black_indices[0],black_indices[1])
    # Iterate over the red points
    v = GS_iteration(red_indices,u,rhs,N-3,h)
    # Iterate over the black points
    z = GS_iteration(black_indices,v,rhs,N-3,h)
    #return the smoothed grid
    return z

def GSRBBR(u,rhs,N,h):
    # u:   the grid to smooth
    # rhs: the right hand side function to use
    # N:   the size of the grid
    # h:   the grid spacing

    # Get a checkerboard of 1s and 0s
    checkerboard = get_checkerboard_of_size(u.shape[0])
    # Gather the indices of the "red" and "black" points
    red_indices = np.where(checkerboard == 1)
    black_indices = np.where(checkerboard == 0)
    red_indices = zip(red_indices[0],red_indices[1])
    black_indices = zip(black_indices[0],black_indices[1])
    # Iterate over the red points
    v = GS_iteration(red_indices,u,rhs,N-3,h)
    # Iterate over the black points
    z = GS_iteration(black_indices,v,rhs,N-3,h)
    # Iterate over the black points
    q = GS_iteration(black_indices,z,rhs,N-3,h)
    # Iterate over the red points
    w = GS_iteration(red_indices,q,rhs,N-3,h)
    #return the smoothed grid
    return w

def GS_lex(u,rhs,N,h):
    inds = []
    for i in xrange(u.shape[0]):
        for j in xrange(u.shape[0]):
            inds.append((i,j))
    v = GS_iteration(inds, u, rhs, N-3,h)
    print inds
    return v

