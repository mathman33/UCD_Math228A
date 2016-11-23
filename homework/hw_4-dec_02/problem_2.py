from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import gc as garbage
from tabulate import tabulate
from math import pi
from tqdm import tqdm
from get_mesh import get_mesh
from V_cycle import V_cycle
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

def pls_plot(X,Y,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1)
    plt.show()
    plt.close()
    garbage.collect()

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
    for i in xrange(5):
        for j in xrange(i+1):
            nus.append((j,i-j))

    # Initialize u and get the right hand side of the equation
    u = np.zeros((2**ARGS.power - 1,2**ARGS.power - 1))
    X,Y,N,h = get_mesh(ARGS.power)
    f = RHS(Y,X)
    real_solution = solution(Y,X)
    e_0 = real_solution - u
    denom = norm(e_0)

    data = []
    for nu in nus:
        u = np.zeros((2**ARGS.power - 1,2**ARGS.power - 1))
        FRACS = []
        for i in xrange(ARGS.max_iterations):
            u_old = u + 0
            u = V_cycle(ARGS.power, u_old, f, nu)
            num = norm(real_solution - u)
            E = u - u_old
            FRACS.append((num/denom)**(1/(i+1)))
            if condition(norm(E),norm(u_old),ARGS.tolerance):
                break

        data.append((nu[0],nu[1],FRACS[0],np.average(FRACS[:2]),np.average(FRACS[:3]),np.average(FRACS[:5]),np.average(FRACS),i+1))
    print tabulate(data, headers=["nu_1", "nu_2", "f_1", "f_2", "f_3", "f_5", "f_max", "its"], tablefmt="latex")





if __name__ == "__main__":
    main()
