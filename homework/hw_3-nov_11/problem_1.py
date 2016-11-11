from __future__ import division

import argparse
import numpy as np
import numpy.linalg as LA
import time

from tqdm import tqdm,tqdm_notebook
from copy import copy,deepcopy
from pprint import pprint


def RHS(x,y):
    return -np.exp(-(x - 0.25)**2 - (y - 0.6)**2)

def do_jacobi(h):
    iteration_limit = 1000000

    tolerance = 0.01

    # Form the mesh
    N = 1/h + 1
    unit = np.linspace(0,1,N,endpoint=True)
    xx,yy = np.meshgrid(unit,unit,sparse=True)

    # Define the function, sampled at the gridpoints
    f = RHS(xx,yy)

    # Initialize u.. why not with the function itself?
    u = deepcopy(f)
    # keep u=0 on the boundaries
    for i in xrange(len(unit)):
        u[i][0] = 0
        u[i][len(unit)-1] = 0
        u[0][i] = 0
        u[len(unit)-1][i] = 0

    relative_errors = []    
    t = tqdm(xrange(iteration_limit))
    for k in t:
        V = np.zeros(shape = (len(unit),len(unit)))
        for i in tqdm(xrange(1,len(unit)-2)):
            for j in tqdm(xrange(1,len(unit)-2)):
                V[i][j] = (1/4)*(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - (h**2)*f[i][j])
        n = LA.norm(u - V,1)/LA.norm(u,1)
        relative_errors.append(n)
        if n < tolerance:
            u = deepcopy(V)
            break
        u = deepcopy(V)
        t.set_description("Jac. It. E=%.3f" % n)

    return u,relative_errors

def do_gs(h):
    iteration_limit = 1000

    # Form the mesh
    N = 1/h + 1
    unit = np.linspace(0,1,N,endpoint=True)
    xx,yy = np.meshgrid(unit,unit,sparse=True)

    # Define the function, sampled at the gridpoints
    F = ((h**2)/4)*RHS(xx,yy)

    # Initialize u.. why not with the function itself?
    u = deepcopy(F)
    # keep u=0 on the boundaries
    for i in xrange(len(unit)):
        u[i][0] = 0
        u[i][len(unit)-1] = 0
        u[0][i] = 0
        u[len(unit)-1][i] = 0

    for k in xrange(iteration_limit):
        V = np.zeros(shape = (len(unit),len(unit)))
        for i in xrange(1,len(unit)-2):
            for j in xrange(1,len(unit)-2):
                u[i][j] = (1/4)*(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - (h**2)*f[i][j])

    return u

def do_sor(h):
    iteration_limit = 1000

    # Form the mesh
    N = 1/h + 1
    unit = np.linspace(0,1,N,endpoint=True)
    xx,yy = np.meshgrid(unit,unit,sparse=True)

    # Define the function, sampled at the gridpoints
    F = ((h**2)/4)*RHS(xx,yy)

    # Initialize u.. why not with the function itself?
    u = deepcopy(F)
    # keep u=0 on the boundaries
    for i in xrange(len(unit)):
        u[i][0] = 0
        u[i][len(unit)-1] = 0
        u[0][i] = 0
        u[len(unit)-1][i] = 0

    # define the relaxation parameter
    omega = 1/4

    for k in xrange(iteration_limit):
        V = np.zeros(shape = (len(unit),len(unit)))
        for i in xrange(1,len(unit)-2):
            for j in xrange(1,len(unit)-2):
                u[i][j] = (omega/4)*(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - (h**2)*f[i][j]) + (1 - omega)*u[i][j]

    return u


def PARSE_ARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jacobi", action="store_true", dest="jacobi", default=False)
    parser.add_argument("-gs", "--gauss_seidel", action="store_true", dest="gauss_seidel", default=False)
    parser.add_argument("-sor", "--sor", action="store_true", dest="sor", default=False)
    parser.add_argument("-m", "--mesh_sizes", type=float, nargs="+", dest="mesh_sizes", required=True)
    return parser.parse_args()

def main():
    ARGS = PARSE_ARGS()

    times = []
    jacobi_solns = []
    relative_errorss = []
    for mesh_size in ARGS.mesh_sizes:
        if ARGS.jacobi:
            tic = time.clock()
            jacobi_soln, relative_errors = do_jacobi(mesh_size)
            toc = time.clock()
            times.append(toc - tic)
            jacobi_solns.append(jacobi_soln)
            relative_errorss.append(relative_errors)
        if ARGS.gauss_seidel:
            gs_data = do_GS(mesh_size)
        if ARGS.sor:
            sor_data = do_sor(mesh_size)

    print "\n\n"
    print times
    print jacobi_solns
    print relative_errorss


if __name__ == "__main__":
    main()