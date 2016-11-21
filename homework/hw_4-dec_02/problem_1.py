from __future__ import division

import numpy as np
from math import log
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gc as garbage


def RHS(X,Y):
    return -np.exp(-(X - 0.25)**2 - (Y - 0.6)**2)

def Dirichlet(u):
    I = u.shape[0]
    J = u.shape[1]
    for i in xrange(I):
        u[i][0] = 0
        u[i][J-1] = 0
    for j in xrange(J):
        u[0][j] = 0
        u[I-1][j] = 0
    return u

def smooth(u):
    pass

def restriction(u):
    power = int(log(u.shape[0]-1,2))-1
    coarse_grid = 2**power+1
    v = np.zeros((coarse_grid,coarse_grid))
    # Don't include edges since we are assuming Dirichlet boundary conditions.
    for i in xrange(1,coarse_grid-1):
        for j in xrange(1,coarse_grid-1):
            row_1 = u[(2*i)-1][(2*j)-1] + 2*u[(2*i)-1][(2*j)] + u[(2*i)-1][(2*j)+1]
            row_2 = 2*(u[(2*i)][(2*j)-1] + 2*u[(2*i)][(2*j)] + u[(2*i)][(2*j)+1])
            row_3 = u[(2*i)+1][(2*j)-1] + 2*u[(2*i)+1][(2*j)] + u[(2*i)+1][(2*j)+1]
            v[i][j] = (1/16)*(row_1 + row_2 + row_3)
    return v

def interpolation(v):
    power = int(log(v.shape[0]-1,2))
    coarse_grid = 2**power+1
    power += 1
    fine_grid = 2**power+1
    u = np.zeros((fine_grid,fine_grid))
    for i in xrange(coarse_grid):
        for j in xrange(coarse_grid):
            # Top row
            if i > 0:
                if j > 0:
                    u[(2*i)-1][(2*j)-1] += (1/4)*v[i][j]
                u[(2*i)-1][(2*j)] += (1/2)*v[i][j]
                if j < coarse_grid-1:
                    u[(2*i)-1][(2*j)+1] += (1/4)*v[i][j]
            # Middle row
            if j > 0:
                u[(2*i)][(2*j)-1] += (1/2)*v[i][j]
            u[(2*i)][(2*j)] += v[i][j]
            if j < coarse_grid-1:
                u[(2*i)][(2*j)+1] += (1/2)*v[i][j]
            # Bottom row
            if i < coarse_grid-1:
                if j > 0:
                    u[(2*i)+1][(2*j)-1] += (1/4)*v[i][j]
                u[(2*i)+1][(2*j)] += (1/2)*v[i][j]
                if j < coarse_grid-1:
                    u[(2*i)+1][(2*j)+1] += (1/4)*v[i][j]
    return u

def correction(u):
    pass

def solve(u):
    pass

def get_red_blue(u):
    red = []

def get_checkerboard_of_size(u):
    red = np.r_[int(u.shape[0]/2)*[1,0] + [1]]
    blue = np.r_[int(u.shape[0]/2)*[0,1] + [0]]
    t = int(u.shape[1]/2)*(red, blue)
    t += red,
    t = np.row_stack(t)
    return t

def GSRB(u,rhs,h):
    checkerboard = get_checkerboard_of_size(u)
    red_indicies = np.where(checkerboard == 1)
    blue_indicies = np.where(checkerboard == 0)
    red_indicies = zip(red_indicies[0],red_indicies[1])
    blue_indicies = zip(blue_indicies[0],blue_indicies[1])
    for (i,j) in red_indicies:
        if i != 0 and i != h-1 and j != 0 and j != h-1:
            u[i][j] = (1/4)*(u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1] - h**2*rhs[i][j])
    for (i,j) in blue_indicies:
        if i != 0 and i != h-1 and j != 0 and j != h-1:
            u[i][j] = (1/4)*(u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1] - h**2*rhs[i][j])
    return u


def get_mesh(power):
    nx = 2**power+1
    ny = 2**power+1
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    X,Y = np.meshgrid(x,y)
    return X,Y,nx

def main():
    power = 6
    X,Y,h = get_mesh(power)
    f = RHS(X,Y)
    u = RHS(X,Y)
    u = Dirichlet(u)
    for k in xrange(30):
        u = GSRB(u,f,h)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,u,rstride=1,cstride=1)
    ax.set_xlabel("X Label")
    plt.show()
    plt.close()
    garbage.collect()

if __name__ == "__main__":
    main()
