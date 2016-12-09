from __future__ import division

from operators import get_laplacian
from get_mesh import get_mesh
from numpy import ravel_multi_index as sub2ind
import scipy.sparse
import numpy as np

def make_matrix_rhs_circleproblem(N):
    # N: grid size

    # Get grid spacing
    dx = 1/(N+1)
    # create a meshgrid
    x = [i*dx for i in xrange(1,N+1)]
    X,Y = np.meshgrid(x,x)
    # Create the 2D discrete Laplacian
    A = (dx**2)*get_laplacian(N)
    # initialize the right hand side
    f = np.zeros(N**2)
    # Define the boundary condition on the circle
    Ub = 1
    # Define the center and radius of the circle
    xc = 0.3
    yc = 0.4
    rad = 0.15
    # precompute the distances of all points on the meshgrid
    # to the center of the circle
    phi = np.sqrt((X - xc)**2 + (Y - yc)**2) - rad
    # Predefine common index additions (up,down,left,right)
    IJ = [[-1,0],[1,0],[0,-1],[0,1]]
    # Loop through the grid
    for j in xrange(1,N-1):
        for i in xrange(1,N-1):
            # Don't do anything inside the circle
            if phi[i][j] < 0:
                continue
            # loop through common indicies (up,down,left,right)
            for k in xrange(4):
                # only update matrix for points on the boundary
                if phi[i + IJ[k][0]][j + IJ[k][1]] < 0:
                    # Get the appropriate distance to the boundary
                    alpha_num = phi[i][j]
                    alpha_den = phi[i][j]-phi[i+IJ[k][0]][j+IJ[k][1]]
                    alpha = phi[i][j]/alpha_den
                    # Get the distance to the boundary
                    kr = sub2ind([i,j],(N,N))
                    kc = sub2ind([i+IJ[k][0],j+IJ[k][1]],(N,N))
                    # adjust the right hand side
                    f[kr] = f[kr] - Ub/alpha
                    # adjust the diagonal entry
                    A[kr,kr] = A[kr,kr] + 1 - 1/alpha
                    # adjust the off-diagonal entries
                    A[kr,kc] = 0
                    A[kc,kr] = 0
    # return the matrix A and the right hand side f
    return (A, f)
