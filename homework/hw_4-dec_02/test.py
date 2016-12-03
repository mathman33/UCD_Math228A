from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import gc as garbage
from tqdm import tqdm
from get_mesh import get_mesh
from V_cycle import V_cycle
from math import log
import copy
from operators import get_laplacian
from smoothers import GS_lex, GSRB
from mpl_toolkits.mplot3d import Axes3D
from time import clock
from argparse import ArgumentParser

HELP = {
    "-m": "Specify maximum number of iterations",
    "-t": "Specify tolerance",
    "-p": "Specify what power of 2 we start our grid",
    "-n": "Specify norm (\"max\" (default), \"one\", or \"two\")",
    "-c": "Specify convergence type (\"abslolute\" or \"relative\" (default))"
}

def RHS(X,Y):
    return np.sin(np.pi*Y)*(10*(np.pi**2)*np.cos(np.pi*X)*np.cos(5*np.pi*X) - 26*np.pi**2*np.sin(np.pi*X)*np.sin(5*np.pi*X)) - (np.pi**2)*np.sin(np.pi*X)*np.sin(5*np.pi*X)*np.sin(np.pi*Y)

def which_norm(s):
    if s == "max":
        def n(a):
            return np.amax(np.abs(a))
    elif s == "one":
        def n(a):
            return sum(np.abs(a.flatten))
    elif s == "two":
        def n(a):
            return sqrt(sum(a.flatten**2))
    return n

def which_condition(s):
    if s == "relative":
        def condition(a,b,tol):
            return a < b*tol
    elif s == "absolute":
        def condition(a,b,tol):
            return a < tol
    return condition

def pls_plot(X,Y,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X,Y,Z,rstride=10,cstride=10)
    # ax.set_zlim(-1,1)
    plt.show()
    plt.close()
    garbage.collect()

def PARSE_ARGS():
    parser = ArgumentParser()
    parser.add_argument("-m", "--max_iterations", type=int, default=100, dest="max_iterations", help=HELP["-m"])
    parser.add_argument("-t", "--tolerance", type=float, default = 10**-7, dest="tolerance", help=HELP["-t"])
    parser.add_argument("-p", "--power", type=int, default=8, dest="power", help=HELP["-p"])
    parser.add_argument("-n", "--norm", type=str, default="max", help=HELP["-n"])
    parser.add_argument("-c", "--convergence", type=str, default="relative", help=HELP["-c"])
    return parser.parse_args()

def main():
    # # get the discrete laplacian of size N-2
    # L = disc_lapl(N-2,h)
    # # flatten the matrices into vectors and set R = f - Lv
    # R = f.flatten() - L.dot(v.flatten())
    # # reshape R into a matrix
    # r = (R).reshape((int(2**power - 1),int(2**power - 1)))

    ARGS = PARSE_ARGS()
    norm = which_norm(ARGS.norm)
    condition = which_condition(ARGS.convergence)
    
    # initialize a guess
    u = np.zeros((2**ARGS.power - 1,2**ARGS.power - 1))
    X,Y,N,h = get_mesh(ARGS.power)
    # f = RHS(X,Y)

    Ls = {}
    for i in xrange(1,10):
        Ls[str(i)] = get_laplacian(2**i - 1)

    # Set the algebraic solution
    ALG_SOL = np.random.rand(2**ARGS.power-1,2**ARGS.power-1)
    # get the discrete laplacian of size N-2
    L = get_laplacian(N-2)
    # flatten the matrices into vectors and set R = f - Lv
    F = Ls[str(int(log(N-1,2)))].dot(ALG_SOL.flatten())
    # reshape R into a matrix
    f = (F).reshape((int(2**ARGS.power - 1),int(2**ARGS.power - 1)))

    t = tqdm(xrange(ARGS.max_iterations))
    for i in t:
        u_old = copy.deepcopy(u)
        u = V_cycle(ARGS.power, u_old, f, (1,1), X, Y, N, h,Ls)
        E = u-u_old
        normE = norm(E)
        #     break
        if condition(normE,norm(u_old),ARGS.tolerance):
            break
        t.set_description("||E||=%.10f" % normE)

    print "\n\niterations = %d" % i

    pls_plot(X,Y,u)




if __name__ == "__main__":
    main()
