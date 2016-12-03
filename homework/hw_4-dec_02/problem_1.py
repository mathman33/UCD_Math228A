from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import gc as garbage
from tqdm import tqdm
from get_mesh import get_mesh
from V_cycle import V_cycle
from operators import compute_residual, get_laplacian
from mpl_toolkits.mplot3d import Axes3D
from time import clock
from argparse import ArgumentParser
from tabulate import tabulate

HELP = {
    "-m": "Specify maximum number of iterations",
    "-t": "Specify tolerance",
    "-p": "Specify what power of 2 we start our grid",
    "-n": "Specify norm (\"max\" (default), \"one\", or \"two\")",
    "-c": "Specify convergence type (\"abslolute\" or \"relative\" (default))",
    "-v": "Specify nu_1 and nu_2 values"
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

def contour(X,Y,Z,its,filename):
    plt.figure()
    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")
    CS = plt.contour(X,Y,Z,10)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(r"approximate solution $u$" + "   %d iterations" % its)
    plt.savefig(filename+".png", type="png", dpi=300)
    plt.close()
    garbage.collect()

def pls_plot(X,Y,Z, its, filename, az, zlabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X,Y,Z,rstride=10,cstride=10)
    ax.view_init(azim=az)
    axx = ax.get_axes()
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.savefig(filename+".png", type="png", dpi=300)
    plt.close()
    garbage.collect()

def PARSE_ARGS():
    parser = ArgumentParser()
    parser.add_argument("-m", "--max_iterations", type=int, default=100, dest="max_iterations", help=HELP["-m"])
    parser.add_argument("-v", "--nus", nargs=2, type=int, default=(1,1), dest="nus", help=HELP["-v"])
    parser.add_argument("-t", "--tolerance", type=float, default = 10**-7, dest="tolerance", help=HELP["-t"])
    parser.add_argument("-p", "--power", type=int, default=8, dest="power", help=HELP["-p"])
    parser.add_argument("-n", "--norm", type=str, default="max", help=HELP["-n"])
    parser.add_argument("-c", "--convergence", type=str, default="relative", help=HELP["-c"])
    return parser.parse_args()

def main():
    ARGS = PARSE_ARGS()
    print "grid size: 2^%d-1" % ARGS.power
    print "tolerance:", ARGS.tolerance
    norm = which_norm(ARGS.norm)
    condition = which_condition(ARGS.convergence)

    Ls = {}
    for i in xrange(1,10):
        Ls[str(i)] = get_laplacian(2**i - 1)

    # Initialize u and get the right hand side of the equation
    u = np.zeros((2**ARGS.power - 1,2**ARGS.power - 1))
    X,Y,N,h = get_mesh(ARGS.power)
    f = RHS(Y,X)
    normf = norm(f)

    t = tqdm(xrange(ARGS.max_iterations))
    residual_norms = []
    for i in t:
        u_old = u + 0
        u = V_cycle(ARGS.power, u_old, f, ARGS.nus, X, Y, N, h, Ls)
        res = compute_residual(u,f,Ls)
        normres = norm(res)
        residual_norms.append(normres)
        if condition(normres,normf,ARGS.tolerance):
            its = i+1
            break
        t.set_description("||res||=%.10f" % normres)

    print "\n\niterations = %d" % its

    ratios = ["-"] + [residual_norms[i]/residual_norms[i-1] for i in xrange(1,len(residual_norms))]
    table = [[i+1, residual_norms[i], ratios[i]] for i in xrange(len(ratios))]
    print tabulate(table, headers=["\$\\norm{r_k}_\infty\$", r"$\frac{||r_k||}{||r_{k-1}||}$"], tablefmt="latex")

    # pls_plot(X, Y, u, its, "figures/p1_run1_1", -60, r"$u$", r"approximate solution $u$" + "   %d iterations" % its)
    # pls_plot(X, Y, u, its, "figures/p1_run1_2", 120, r"$u$", r"approximate solution $u$" + "   %d iterations" % its)
    # pls_plot(X, Y, np.abs(res), its, "figures/p1_run1_3", -60, r"$|r|$", r"abs. val. residual $|r|$" + "   %d iterations" % its)
    # contour(X, Y, u, its, "figures/p1_run1_4")
    # contour(X, Y, res, its, "figures/p1_run1_5")




if __name__ == "__main__":
    main()
