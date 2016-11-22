from __future__ import division

import scipy.sparse
import scipy.sparse.linalg
from copy import deepcopy
import numpy as np
from math import log
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gc as garbage


def RHS(X,Y):
    return -np.exp(-(X - 0.25)**2 - (Y - 0.6)**2)

def max_norm(a):
    return np.amax(np.abs(a))

def restriction(u):
    power = int(log(u.shape[0]+1,2))-1
    coarse_grid = 2**power-1
    v = np.zeros((coarse_grid,coarse_grid))
    for i in xrange(0,coarse_grid):
        for j in xrange(0,coarse_grid):
            row_1 = u.item(((2*i),(2*j))) + 2*u.item(((2*i),(2*j)+1)) + u.item(((2*i),(2*j)+2))
            row_2 = 2*(u.item(((2*i)+1,(2*j))) + 2*u.item(((2*i)+1,(2*j)+1)) + u.item(((2*i)+1,(2*j)+2)))
            row_3 = u.item(((2*i)+2,(2*j))) + 2*u.item(((2*i)+2,(2*j)+1)) + u.item(((2*i)+2,(2*j)+2))
            v[i][j] = (1/16)*(row_1 + row_2 + row_3)
    return v

def interpolation(v,power):
    coarse_grid = 2**(power-1)-1
    fine_grid = 2**power-1
    u = np.zeros((fine_grid,fine_grid))
    for i in xrange(coarse_grid):
        for j in xrange(coarse_grid):
            # Top row
            u[(2*i)][(2*j)] += (1/4)*v[i][j]
            u[(2*i)][(2*j)+1] += (1/2)*v[i][j]
            u[(2*i)][(2*j)+2] += (1/4)*v[i][j]

            # Middle row
            u[(2*i)+1][(2*j)] += (1/2)*v[i][j]
            u[(2*i)+1][(2*j)+1] += v[i][j]
            u[(2*i)+1][(2*j)+2] += (1/2)*v[i][j]

            # Bottom row
            u[(2*i)+2][(2*j)] += (1/4)*v[i][j]
            u[(2*i)+2][(2*j)+1] += (1/2)*v[i][j]
            u[(2*i)+2][(2*j)+2] += (1/4)*v[i][j]
    return u

def get_red_blue(u):
    red = []

def get_checkerboard_of_size(u):
    red = np.r_[int(u.shape[0]/2)*[1,0] + [1]]
    blue = np.r_[int(u.shape[0]/2)*[0,1] + [0]]
    t = int(u.shape[1]/2)*(red, blue)
    t += red,
    t = np.row_stack(t)
    return t

def GSRB(u,rhs,N,h):
    checkerboard = get_checkerboard_of_size(u)
    red_indicies = np.where(checkerboard == 1)
    blue_indicies = np.where(checkerboard == 0)
    red_indicies = zip(red_indicies[0],red_indicies[1])
    blue_indicies = zip(blue_indicies[0],blue_indicies[1])
    for indices in [red_indicies, blue_indicies]:
        for (i,j) in indices:
            if i == 0:
                first = 0
            else:
                first = u[i-1][j]
            if j == 0:
                second = 0
            else:
                second = u[i][j-1]
            if i == N-3:
                third = 0
            else:
                third = u[i+1][j]
            if j == N-3:
                fourth = 0
            else:
                fourth = u[i][j+1]
            u[i][j] = (1/4)*(first + second + third + fourth - (h**2)*rhs[i][j])
    return u


def pls_plot(X,Y,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1)
    plt.show()
    plt.close()
    garbage.collect()

def get_mesh(power):
    N = 2**power+1
    x,h = np.linspace(0,1,N,retstep=True)
    x = x[1:len(x)-1:]
    X,Y = np.meshgrid(x,x)
    return X,Y,N,h

def disc_lapl(N,h):
    off_diag = 1/(h**2)*np.ones(N**2)
    diag = -4/(h**2)*np.ones(N**2)
    A = np.vstack((off_diag,off_diag,diag,off_diag,off_diag))
    A = scipy.sparse.dia_matrix((A,[-N,-1,0,1,N]),shape=(N**2,N**2))
    A = scipy.sparse.csr_matrix(A)
    return A

def trivial_direct_solve(h,r):
    return np.asarray((-(h**2)/4)*r.item((0,0))).reshape((1,1))

def V_cycle(power,u,f):
    X,Y,N,h = get_mesh(power)

    # Smooth
    u = GSRB(u,f,N,h)
    u = u.transpose()

    # Compute Residual
    L = disc_lapl(N-2,h)
    r = (f.transpose().flatten() - L.dot(u.flatten())).reshape((int(2**power - 1),int(2**power - 1)))

    # Restrict
    r = restriction(r)

    # Solve
    if power == 2:
        e = trivial_direct_solve(h,r)
    else:
        e_guess = np.zeros((2**(power-1) - 1, 2**(power-1) - 1))
        e = V_cycle(power-1,e_guess,r)

    # Interpolate the error
    e = interpolation(e,power)

    # Correct the solution
    u = u + e

    # Smooth
    u = GSRB(u,f,N,h)
    u = u.transpose()

    return u


def main():
    tolerance = 10**-3
    power = 2
    u = np.zeros((2**power - 1,2**power - 1))
    X,Y,N,h = get_mesh(power)
    while True:
        u_new = V_cycle(power, u, RHS(X,Y))
        E = u_new-u
        print max_norm(E)/max_norm(u)
        u = deepcopy(u_new)
        if max_norm(E)/max_norm(u) < tolerance:
            break





if __name__ == "__main__":
    main()
