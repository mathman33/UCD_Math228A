from __future__ import division

import argparse
import numpy as np
import numpy.linalg as LA
import time
import scipy.sparse
import scipy.sparse.linalg
import pickle

from tqdm import tqdm,tqdm_notebook
from copy import copy,deepcopy

iteration_limit = 1000000

def RHS(x,y):
    return -np.exp(-(x - 0.25)**2 - (y - 0.6)**2)

def do_direct(h,ARGS):
    N = 1/h + 1
    unit = np.linspace(0,1,N,endpoint=True)
    xx,yy = np.meshgrid(unit,unit,sparse=True)
    f = RHS(xx,yy)
    f = f.flatten()

    sub_diag = (1/h**2)*np.ones(N**2)
    super_diag = (1/h**2)*np.ones(N**2)
    supersuper_diag = (1/h**2)*np.ones(N**2)
    subsub_diag = (1/h**2)*np.ones(N**2)
    diag = (-4/h**2)*np.ones(N**2)

    A = np.vstack((subsub_diag,sub_diag,diag,super_diag,supersuper_diag))
    A = scipy.sparse.dia_matrix((A,[-N,-1,0,1,N]),shape=(N**2,N**2))
    A = scipy.sparse.csc_matrix(A)

    print "inverting.."
    soln = scipy.sparse.linalg.spsolve(A,f)
    print "inverted"
    return soln

def do_jacobi(h,ARGS):
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
            for j in xrange(1,len(unit)-2):
                V[i][j] = (1/4)*(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - (h**2)*f[i][j])
        n = LA.norm(u - V,1)/LA.norm(u,1)
        relative_errors.append(n)
        if n < ARGS.tolerance:
            u = deepcopy(V)
            break
        u = deepcopy(V)
        t.set_description("Jac. It. E=%.5f" % n)

    return u,relative_errors

def do_gs(h, ARGS):
    # Form the mesh
    N = 1/h + 1
    unit = np.linspace(0,1,N,endpoint=True)
    xx,yy = np.meshgrid(unit,unit,sparse=True)

    # Define the function, sampled at the gridpoints
    f = RHS(xx,yy)

    # Initialize u.. why not with the function itself?
    u = deepcopy(f)

    # keep u=0 on the boundaries
    for i in tqdm(xrange(len(unit)),desc="bdry conds."):
        u[i][0] = 0
        u[i][len(unit)-1] = 0
        u[0][i] = 0
        u[len(unit)-1][i] = 0

    relative_errors = []
    t = tqdm(xrange(iteration_limit))
    for k in t:
        U = deepcopy(u)
        for i in tqdm(xrange(1,len(unit)-2)):
            for j in xrange(1,len(unit)-2):
                u[i][j] = (1/4)*(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - (h**2)*f[i][j])
        n = LA.norm(u - U,1)/LA.norm(u,1)
        relative_errors.append(n)
        if n < ARGS.tolerance:
            break
        t.set_description("GS. It. E=%.5f" % n)

    return u,relative_errors

def do_sor(h,ARGS):
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

    # define the relaxation parameter
    omega = 2*(1 - np.pi*h)

    relative_errors = []
    t = tqdm(xrange(iteration_limit))
    for k in t:
        U = deepcopy(u)
        for i in tqdm(xrange(1,len(unit)-2)):
            for j in xrange(1,len(unit)-2):
                u[i][j] = (omega/4)*(u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - (h**2)*f[i][j]) + (1 - omega)*u[i][j]
        n = LA.norm(u - U,1)/LA.norm(u,1)
        relative_errors.append(n)
        if n < ARGS.tolerance:
            break
        t.set_description("SOR. It. E=%.5f" % n)

    return u,relative_errors


def PARSE_ARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jacobi", action="store_true", dest="jacobi", default=False)
    parser.add_argument("-gs", "--gauss_seidel", action="store_true", dest="gauss_seidel", default=False)
    parser.add_argument("-sor", "--sor", action="store_true", dest="sor", default=False)
    parser.add_argument("-d", "--direct", action="store_true", dest="direct", default=False)
    parser.add_argument("-m", "--mesh_sizes", type=float, nargs="+", dest="mesh_sizes", required=True)
    parser.add_argument("-t", "--tolerance", type=float, nargs=1, dest="tolerance", default=0.01, required=True)
    return parser.parse_args()

def main():
    ARGS = PARSE_ARGS()

    jacobi_data = {
        "times":[],
        "solns":[],
        "rel_ers":[]
    }
    gs_data = {
        "times":[],
        "solns":[],
        "rel_ers":[]
    }
    sor_data = {
        "times":[],
        "solns":[],
        "rel_ers":[]
    }
    direct_data = {
        "times":[],
        "solns":[],
    }
    jacobi_times = []
    jacobi_solns = []
    jacobi_relative_errors = []
    if ARGS.jacobi:
        for mesh_size in ARGS.mesh_sizes:
            tic = time.clock()
            jacobi_soln, relative_errors = do_jacobi(mesh_size,ARGS)
            toc = time.clock()
            jacobi_data["times"].append(toc - tic)
            jacobi_data["solns"].append(jacobi_soln)
            jacobi_data["rel_ers"].append(relative_errors)
        with open("jacobi_data_3.p", "wb") as handle:
            pickle.dump(jacobi_data, handle)
    if ARGS.gauss_seidel:
        for mesh_size in ARGS.mesh_sizes:
            tic = time.clock()
            gs_soln, relative_errors = do_gs(mesh_size,ARGS)
            toc = time.clock()
            gs_data["times"].append(toc - tic)
            gs_data["solns"].append(gs_soln)
            gs_data["rel_ers"].append(relative_errors)
        with open("gs_data_3.p", "wb") as handle:
            pickle.dump(gs_data, handle)
    if ARGS.sor:
        for mesh_size in ARGS.mesh_sizes:
            tic = time.clock()
            sor_soln, relative_errors = do_sor(mesh_size,ARGS)
            toc = time.clock()
            sor_data["times"].append(toc - tic)
            sor_data["solns"].append(sor_soln)
            sor_data["rel_ers"].append(relative_errors)
        with open("sor_data_3.p", "wb") as handle:
            pickle.dump(sor_data, handle)
    if ARGS.direct:
        for mesh_size in ARGS.mesh_sizes:
            tic = time.clock()
            direct_soln = do_direct(mesh_size,ARGS)
            toc = time.clock()
            direct_data["times"].append(toc - tic)
            direct_data["solns"].append(direct_soln)
        with open("direct_data_3.p", "wb") as handle:
            pickle.dump(direct_data, handle)

if __name__ == "__main__":
    main()