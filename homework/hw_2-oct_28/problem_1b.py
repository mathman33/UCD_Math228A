from __future__ import division


import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import tabulate


def solution(x,N):
    mean_zero_sol = 0.5*(x**2) - (1/(4*(np.pi**2)))*np.cos(2*np.pi*x) - (1/6)
    sol = mean_zero_sol - (1/(N+2))*sum(mean_zero_sol)
    return sol

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

    sub_diag = np.concatenate((1/(h**2)*np.ones(N), [2/(h**2),0]))
    super_diag = np.concatenate(([0,2/(h**2)], 1/(h**2)*np.ones(N)))
    diag = -2/(h**2)*np.ones(N+2)

    A = np.vstack((sub_diag,diag,super_diag))
    A = scipy.sparse.dia_matrix((A,[-1,0,1]),shape=(N+2,N+2))
    A = scipy.sparse.hstack([A, np.append([1], np.append(2*np.ones((N,1)),[1])).reshape(N+2,1)])
    A = scipy.sparse.vstack([A, np.concatenate((np.ones(N+2),[0]))])
    A = scipy.sparse.csc_matrix(A)

    grid_pts = np.linspace(0, 1, num=(N+2))
    b = 2*(np.cos(np.pi*grid_pts)**2)
    b[-1] = b[-1] - (2/h)

    u = scipy.sparse.linalg.spsolve(A, np.concatenate((b, [0])))
    u = np.delete(u,N+2)

    tru_sol = solution(grid_pts, N)

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
# plt.title(r"Absolute Errors vs. Grid Size   $u_{xx} = 2\cos^2(\pi x)$, $u_x(0) = 0$, $u_x(1) = 1$")
# plt.legend(loc=0)
# plt.savefig("figure_1b.png", filetype="png", dpi=300)
# plt.close()

