from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import gc as garbage
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
    return -np.exp(-(X - 0.25)**2 - (Y - 0.6)**2)

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
    parser.add_argument("-t", "--tolerance", type=float, default = 10**-7, dest="tolerance", help=HELP["-t"])
    parser.add_argument("-p", "--power", type=int, default=8, dest="power", help=HELP["-p"])
    parser.add_argument("-n", "--norm", type=str, default="max", help=HELP["-n"])
    parser.add_argument("-c", "--convergence", type=str, default="relative", help=HELP["-c"])
    return parser.parse_args()

def main():
    ARGS = PARSE_ARGS()
    norm = which_norm(ARGS.norm)
    condition = which_condition(ARGS.convergence)
    
    # Initialize u and get the right hand side of the equation
    u = np.zeros((2**ARGS.power - 1,2**ARGS.power - 1))
    X,Y,N,h = get_mesh(ARGS.power)
    f = RHS(Y,X)

    t = tqdm(xrange(ARGS.max_iterations))
    for i in t:
        u_old = u + 0
        u = V_cycle(ARGS.power, u_old, f, (1,1))
        E = u-u_old
        if condition(norm(E),norm(u_old),ARGS.tolerance):
            break
        t.set_description("||E||=%.10f" % norm(E))

    print "\n\niterations = %d" % i




if __name__ == "__main__":
    main()
