from __future__ import division


import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import math
import matplotlib.pyplot as plt
import tabulate


def solution(x):
    return np.exp(x) + (2 - math.exp(1))*x - 1

def one_norm_error(a,b,h):
    S = 0
    for i in range(len(a)):
        S += abs(a[i]-b[i])
    return S*h

def max_norm_error(a,b):
    S = 0
    for i in range(len(a)):
        S = max(S, abs(a[i]-b[i]))
    return S

one = []
Max = []

Ns = [2**i for i in range(1,11)]

for N in Ns:
    h = 1/(1 + N)
    sub_diag = 1/(h**2)*np.ones(N)
    super_diag = 1/(h**2)*np.ones(N)
    diag = -2/(h**2)*np.ones(N)

    A = np.vstack((sub_diag,diag,super_diag))
    A = scipy.sparse.dia_matrix((A,[-1,0,1]),shape=(N,N))
    A = scipy.sparse.csc_matrix(A)

    grid_pts = np.linspace(h, 1-h, num=N)
    b = np.exp(grid_pts)
    b[-1] -= 1/(h**2)

    u = scipy.sparse.linalg.spsolve(A, b)

    tru_sol = solution(grid_pts)

    e_1 = one_norm_error(tru_sol,u,h)
    e_m = max_norm_error(tru_sol,u)

    one.append(e_1)
    Max.append(e_m)

one_rat = [0]
Max_rat = [0]
for i in range(9):
    one_rat.append(one[i+1]/one[i])
    Max_rat.append(Max[i+1]/Max[i])

thing = [[Ns[i],one[i],one_rat[i],Max[i],Max_rat[i]] for i in xrange(10)]
print tabulate.tabulate(thing, headers=["N","one-norm","one-norm ratios","max norm", "max norm ratios"], tablefmt="latex")

# plt.figure()
# plt.semilogy(Ns, one, "--", lw=2, label=r"$||\cdot||_1$")
# plt.semilogy(Ns, Max, lw=2, label=r"$||\cdot||_\infty$")
# plt.xlabel(r"Interior Grid Points $N$")
# plt.ylabel("Absolute Error")
# plt.title(r"Absolute Errors vs. Grid Size    $u_{xx} = \exp(x)$, $u(0) = 0$, $u(1) = 1$")
# plt.legend(loc=0)
# plt.savefig("figure_1a.png", filetype="png", dpi=300)
# plt.close()

