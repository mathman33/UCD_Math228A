from __future__ import division

import numpy as np
from math import log
import scipy
import scipy.sparse

def restriction(u,power):
    # u:     the fine mesh to restrict
    # power: the power of 2 of the fine mesh

    # get the size of the coarse grid
    coarse_grid = 2**(power-1)-1
    # initialize the coarse grid
    v = np.zeros((coarse_grid,coarse_grid))
    # loop through the coarse grid
    for i in xrange(0,coarse_grid):
        for j in xrange(0,coarse_grid):
            # gather data from the first row in the fine grid
            row_1 = u.item(((2*i),(2*j)))
            row_1 += 2*u.item(((2*i),(2*j)+1))
            row_1 += u.item(((2*i),(2*j)+2))
            # gather data from the second row in the fine grid
            row_2 = 2*u.item(((2*i)+1,(2*j)))
            row_2 += 4*u.item(((2*i)+1,(2*j)+1))
            row_2 += 2*u.item(((2*i)+1,(2*j)+2))
            # gather data from the third row in the fine grid
            row_3 = u.item(((2*i)+2,(2*j)))
            row_3 += 2*u.item(((2*i)+2,(2*j)+1))
            row_3 += u.item(((2*i)+2,(2*j)+2))
            # combine data for coarse grid
            v[i][j] = (1/16)*(row_1 + row_2 + row_3)
    # return the coarse grid
    return v

def interpolation(v,power):
    # v:     the coarse mesh to interpolate
    # power: the power of 2 of the fine mesh

    # get the size of the coarse and fine grids
    coarse_grid = 2**(power-1)-1
    fine_grid = 2**power-1
    # initialize the fine grid
    U = np.zeros((fine_grid,fine_grid))
    # loop through the coarse grid
    for i in xrange(coarse_grid):
        for j in xrange(coarse_grid):
            # add coarse grid data to the first row of the fine grid
            U[(2*i)][(2*j)] += (1/4)*v[i][j]
            U[(2*i)][(2*j)+1] += (1/2)*v[i][j]
            U[(2*i)][(2*j)+2] += (1/4)*v[i][j]
            # add coarse grid data to the second row of the fine grid
            U[(2*i)+1][(2*j)] += (1/2)*v[i][j]
            U[(2*i)+1][(2*j)+1] += v[i][j]
            U[(2*i)+1][(2*j)+2] += (1/2)*v[i][j]
            # add coarse grid data to the third row of the fine grid
            U[(2*i)+2][(2*j)] += (1/4)*v[i][j]
            U[(2*i)+2][(2*j)+1] += (1/2)*v[i][j]
            U[(2*i)+2][(2*j)+2] += (1/4)*v[i][j]
    #return the fine grid
    return U

def Lap_1D(N):
    # N: size of the 1D discrete Laplacian

    # define the Laplacian as the tridiagonal matrix with 1s
    # on the super- and sub-diagonals and -2 on the diagonal
    off_diag = 1*np.ones(N)
    diag = (-2)*np.ones(N)
    A = np.vstack((off_diag,diag,off_diag))
    Laplacian_1D = scipy.sparse.dia_matrix((A,[-1,0,1]),shape=(N,N))
    # return the Laplacian
    return Laplacian_1D

def get_laplacian(N):
    # N: N^2 x N^2 is the size of the 2D Laplacian

    # get the spacing
    h = 1/(N+1)
    # Get the 1D Laplacian
    Laplacian_1D = Lap_1D(N)
    # Get an NxN identity matrix
    I = scipy.sparse.identity(N)
    # Get kronecker products
    kron_prod_1 = scipy.sparse.kron(Laplacian_1D,I)
    kron_prod_2 = scipy.sparse.kron(I,Laplacian_1D)
    # Cleverly define the 2D Laplacian as the sum of Kronecker products
    Laplacian_2D = (1/(h**2))*(kron_prod_1 + kron_prod_2)
    # return the 2D Laplacian
    return Laplacian_2D

def compute_residual(u,f,Ls):
    # u: approximate solution
    # f: right-hand-side of Au = f

    # Get the Laplacian of the appropriate size
    A = Ls[str(int(log(u.shape[0]+1,2)))]
    # Flatten u
    uflat = u.flatten()
    # Compute Au and reshape it back into the appropriate shape
    Au = A.dot(uflat).reshape(u.shape)
    # define the residual as f - Au
    R = f - Au
    # return the residual
    return R

