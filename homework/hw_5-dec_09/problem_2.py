from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import smoothers
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse import csc_matrix as csc
from operators import get_laplacian, compute_residual
from make_matrix_rhs_circleproblem import make_matrix_rhs_circleproblem as mmrc
from tqdm import tqdm
from V_cycle import V_cycle
from get_mesh import get_mesh
from copy import deepcopy
from time import clock,sleep
from argparse import ArgumentParser
# from tabulate import tabulate

HELP = {
    "-p": "The power of 2",
    "-s": "do preconditioning",
    "-S": "do SOR",
    "-m": "maximum PCG iterations",
    "-t": "relative tolerance",
}

def RHS(X,Y):
    return -np.exp(-(X - 0.25)**2 - (Y - 0.6)**2)

def PCG(u,p,z,r,A,ARGS,X,Y,N,h,Ls,normf,As):
    # iterate for "max_iterations" or until the condition is met.
    t = tqdm(xrange(ARGS.max_iterations))
    for i in t:
        # copy the old variables
        u_old = deepcopy(u)
        p_old = deepcopy(p)
        z_old = deepcopy(z)
        r_old = deepcopy(r)
        # get w = Ap
        w = A.dot(p_old.flatten()).reshape(N-2,N-2)
        # get alpha = z.r/p.w
        alpha_num = np.dot(z_old.flatten(),r_old.flatten())
        alpha_denom = np.dot(p_old.flatten(),w.flatten())
        alpha = alpha_num/alpha_denom
        # update u = u + alpha.p
        u = u_old + alpha*p_old
        # update r = r - alpha.w
        r = r_old - alpha*w
        # calculate ||r||
        normr = np.amax(np.abs(r))
        # check the exit condition
        if normr <= normf*ARGS.tol:
            # if the condition is met, break out of the loop
            break
        # Set z as the approximate solution of Mz = r, where
        # M is the discrete Laplacian.
        # z is either one iteration of multigrid, one
        # iteration of SSOR, or simply r
        if ARGS.shampoo == "MG":
            z = V_cycle(ARGS.power,np.zeros(z_old.shape),r,(1,1),X,Y,N,h,As)
        if ARGS.shampoo == "SSOR":
            z = SSOR(r,N,h)
        elif ARGS.shampoo == "none":
            z = deepcopy(r)
        # Set beta = z_k.r_k/z_{k-1}.r_{k-1}
        beta_num = np.dot(z.flatten(),r.flatten())
        beta_denom = np.dot(z_old.flatten(),r_old.flatten())
        beta = beta_num/beta_denom
        # update p = z + beta.p
        p = z + beta*p_old
        # update python output display with the norm of the residual
        t.set_description("||res||=%.10f" % normr)

    # return the solution and number of iterations
    return u, i+1

def SSOR(r, N, h):
    # r: the righthand side of the Laplacian
    # N: The number of grid points
    # h: the grid spacing

    # Calculate the optimal omega
    omega = 2*(1 - np.pi*h)
    # pad the matrix with 0s since my code doesn't take
    # the 0 boundary into account.  It makes the loops easier.
    r = np.pad(r,((1,1),(1,1)),mode="constant")
    # initialize the new matrix (initial guess = 0)
    u = np.zeros(np.shape(r))
    # iterate going forward
    for i in xrange(1, N-1):
        for j in xrange(1,N-1):
            # update the matrix
            UDLR = u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1]
            update = UDLR - (h**2)*r[i][j]
            u[i][j] = (omega/4)*update + (1 - omega)*u[i][j]
    # iterate going backwards
    for i in xrange(N-2,0,-1):
        for j in xrange(N-2,0,-1):
            # update the matrix
            UDLR = u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1]
            update = UDLR - (h**2)*r[i][j]
            u[i][j] = (omega/4)*update + (1 - omega)*u[i][j]
    # return the matrix without the padding
    return u[1:-1, 1:-1]

def SOR(u, r, N, h):
    # r: the righthand side of the Laplacian
    # N: The number of grid points
    # h: the grid spacing

    # Calculate the optimal omega
    omega = 2*(1 - np.pi*h)
    # pad the matrix with 0s since my code doesn't take
    # the 0 boundary into account.  It makes the loops easier.
    r = np.pad(r,((1,1),(1,1)),mode="constant")
    u = np.pad(u,((1,1),(1,1)),mode="constant")
    # iterate through the points in the matrix
    for i in xrange(1, N-1):
        for j in xrange(1,N-1):
            # update the matrix
            UDLR = u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1]
            update = UDLR - (h**2)*r[i][j]
            u[i][j] = (omega/4)*update + (1 - omega)*u[i][j]
    # return the matrix without the padding
    return u[1:-1, 1:-1]

def PARSE_ARGS():
    parser = ArgumentParser()
    parser.add_argument("-p", "--power", type=int, default=7, dest="power", help=HELP["-p"])
    parser.add_argument("-s", "--shampoo", type=str, default="none", dest="shampoo", help=HELP["-s"])
    parser.add_argument("-m", "--max_iterations", type=int, default=1000, dest="max_iterations", help=HELP["-m"])
    parser.add_argument("-t", "--tolerance", type=float, default=1e-7, dest="tol", help=HELP["-t"])
    parser.add_argument("-S", "--SOR_TOO", default=False, action="store_true", dest="SOR_TOO", help=HELP["-S"])
    return parser.parse_args()

def main():
    # Parse command line arguments
    ARGS = PARSE_ARGS()
    print "power: ", ARGS.power
    print "tolerance: ", ARGS.tol
    print "precondition: ", ARGS.shampoo

    # define the mesh X,Y and spacing h
    X,Y,N,h = get_mesh(ARGS.power)

    Ss = {}
    fs = {}
    As = {}
    for i in xrange(1,ARGS.power+1):
        As[str(i)] = get_laplacian(2**i - 1)
    A = As[str(ARGS.power)]
    for i in xrange(1,ARGS.power+1):
        (A_spec, f_spec) = mmrc(int(2**i - 1))
        Ss[str(i)] = (1/(h**2))*A_spec
        fs[str(i)] = f_spec.reshape(int(2**i - 1),int(2**i - 1))
    S = Ss[str(ARGS.power)]
    f = fs[str(ARGS.power)]

    normf = np.amax(np.abs(f))

    # initialize u
    u = np.zeros((N-2,N-2))

    # if ARGS.SOR_TOO:
    #     t = tqdm(xrange(ARGS.max_iterations))
    #     for SOR_ITS in t:
    #         u_old = deepcopy(u)
    #         u = SOR(u_old, f, N, h)
    #         r = f - A.dot(u.flatten()).reshape(N-2,N-2)
    #         normr = np.amax(np.abs(r))
    #         if normr <= ARGS.tol*normf:
    #             SOR_ITS += 1
    #             break
    #         t.set_description("||res||=%.10f" % normr)
    #     print "SOR_ITS: ", SOR_ITS

    # initialize r by calculating the residual f - Au
    r = f - S.dot(u.flatten()).reshape(N-2,N-2)
    # solve Mz = r
    if ARGS.shampoo == "MG":
        z = V_cycle(ARGS.power,u,r,(1,1),X,Y,N,h,As)
    elif ARGS.shampoo == "SSOR":
        z = SSOR(r, N, h)
    elif ARGS.shampoo == "none":
        z = deepcopy(r)

    # initialize p
    p = deepcopy(z)
    # Do PCG
    U,its = PCG(u,p,z,r,S,ARGS,X,Y,N,h,Ss,normf,As)
    print ARGS.shampoo, "ITS: ", its

if __name__ == "__main__":
    main()
