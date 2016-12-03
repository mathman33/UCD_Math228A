from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import gc as garbage
from tabulate import tabulate
from math import pi,log
from tqdm import tqdm
from get_mesh import get_mesh
from V_cycle import V_cycle
from mpl_toolkits.mplot3d import Axes3D
from operators import get_laplacian,compute_residual
from time import clock
from argparse import ArgumentParser
from scipy.stats.mstats import gmean
from time import clock

HELP = {
    "-m": "Specify maximum number of iterations",
    "-t": "Specify tolerance",
    "-p": "Specify what power of 2 we start our grid",
    "-n": "Specify norm (\"max\" (default), \"one\", or \"two\")",
    "-c": "Specify convergence type (\"abslolute\" or \"relative\" (default))"
}

def RHS(X,Y):
    return -2*(pi**2)*np.sin(pi*X)*np.sin(pi*Y)

def solution(X,Y):
    return np.sin(pi*X)*np.sin(pi*Y)

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

def PARSE_ARGS():
    parser = ArgumentParser()
    parser.add_argument("-m", "--max_iterations", type=int, default=100, dest="max_iterations", help=HELP["-m"])
    parser.add_argument("-t", "--tolerance", type=float, default = 10**-6, dest="tolerance", help=HELP["-t"])
    parser.add_argument("-p", "--power", type=int, default=5, dest="power", help=HELP["-p"])
    parser.add_argument("-n", "--norm", type=str, default="max", help=HELP["-n"])
    parser.add_argument("-c", "--convergence", type=str, default="relative", help=HELP["-c"])
    return parser.parse_args()

def main():
    ARGS = PARSE_ARGS()
    norm = which_norm(ARGS.norm)
    condition = which_condition(ARGS.convergence)

    nus = []
    for i in xrange(1,5):
        for j in xrange(i+1):
            nus.append((j,i-j))

    Ls = {}
    for i in xrange(1,10):
        Ls[str(i)] = get_laplacian(2**i - 1)

    # Initialize u and get the right hand side of the equation
    u = np.zeros((2**ARGS.power - 1,2**ARGS.power - 1))
    X,Y,N,h = get_mesh(ARGS.power)
    real_solution = np.random.rand(N-2,N-2)
    f = Ls[str(int(log(N-1,2)))].dot(real_solution.flatten()).reshape(real_solution.shape)
    normf = norm(f)
    e_0 = real_solution - u
    denom = norm(e_0)

    data = []
    for nu in nus:
        tic = clock()
        u = np.zeros((2**ARGS.power - 1,2**ARGS.power - 1))
        FRACS = []
        for i in tqdm(xrange(ARGS.max_iterations),desc="(%d,%d)" % (nu[0],nu[1])):
            u_old = u + 0
            u = V_cycle(ARGS.power, u_old, f, nu, X, Y, N, h, Ls)
            res = compute_residual(u,f,Ls)
            normres = norm(res)
            num = norm(real_solution - u)
            FRACS.append((num/denom)**(1/(i+1)))
            if condition(normres,normf,ARGS.tolerance):
                break
        toc = clock()

        data.append((nu[0],nu[1],FRACS[0],FRACS[1],FRACS[2],FRACS[3],FRACS[4],i+1,toc-tic))
    print "\n\n"
    print tabulate(data, headers=["nu_1", "nu_2", "E_1", "E_2", "E_3", "E_4", "E_5", "its", "time"], tablefmt="latex")





if __name__ == "__main__":
    main()
