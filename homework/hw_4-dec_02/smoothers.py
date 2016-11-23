from __future__ import division

import numpy as np
from copy import deepcopy

def get_checkerboard_of_size(u):
    red = np.r_[int(u.shape[0]/2)*[1,0] + [1]]
    blue = np.r_[int(u.shape[0]/2)*[0,1] + [0]]
    t = int(u.shape[1]/2)*(red, blue)
    t += red,
    t = np.row_stack(t)
    return t

def GSRB_its(inds, u, rhs, N, h):
    v = deepcopy(u)
    for (i,j) in inds:
        if i == 0:
            first = 0
        else:
            first = u[i-1][j]
        if j == 0:
            second = 0
        else:
            second = u[i][j-1]
        if i == N:
            third = 0
        else:
            third = u[i+1][j]
        if j == N:
            fourth = 0
        else:
            fourth = u[i][j+1]
        v[i][j] = (1/4)*(first + second + third + fourth - (h**2)*rhs[i][j])
    return v

def GSRB(u,rhs,N,h):
    checkerboard = get_checkerboard_of_size(u)
    red_indicies = np.where(checkerboard == 1)
    blue_indicies = np.where(checkerboard == 0)
    red_indicies = zip(red_indicies[0],red_indicies[1])
    blue_indicies = zip(blue_indicies[0],blue_indicies[1])
    v = GSRB_its(red_indicies,u,rhs,N-3,h)
    z = GSRB_its(blue_indicies,v,rhs,N-3,h)
    return z