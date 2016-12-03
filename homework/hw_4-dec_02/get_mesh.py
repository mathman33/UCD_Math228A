from __future__ import division

import numpy as np

def get_mesh(power):
    # power: the power of two to use

    # get the grid size (including boundaries)
    N = 2**power+1
    # get equally spaced grid between 0 and 1 (inclusive) and step size
    x,h = np.linspace(0,1,N,retstep=True)
    # omit 0 and 1
    x = x[1:len(x)-1:]
    # make the mesh
    X,Y = np.meshgrid(x,x)
    # return the mesh, number of grid points, and gridsize
    return X,Y,N,h
