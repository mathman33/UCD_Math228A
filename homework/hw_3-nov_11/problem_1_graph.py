from __future__ import division

import argparse
import numpy as np
import numpy.linalg as LA
import time
import pickle
import subprocess
import gc as garbage
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from tqdm import tqdm,tqdm_notebook
from copy import copy,deepcopy







def main():
    with open("jacobi_data.p", "r") as handle:
        jacobi_data = pickle.load(handle)
    with open("gs_data.p", "r") as handle:
        gs_data = pickle.load(handle)
    with open("sor_data.p", "r") as handle:
        sor_data = pickle.load(handle)

    hs = [0.03125, 0.015625, 0.0078125]
    labels = [r"$2^{-5}$",r"$2^{-6}$",r"$2^{-7}$"]
    datas = [jacobi_data,gs_data,sor_data]
    names = ["JACOBI","GS","SOR"]

    for j in xrange(len(datas)):
        data = datas[j]
        # plt.figure()
        for i in xrange(3):
            t = data["times"][i]
            s = data["solns"][i]
            r = data["rel_ers"][i]
            h = hs[i]

            print t

            # plt.loglog(r, label=labels[i])

            # N = 1/h + 1
            # unit = np.linspace(0,1,N,endpoint=True)
            # xx,yy = np.meshgrid(unit,unit,sparse=True)
            # fig = plt.figure()
            # ax = fig.gca(projection="3d")
            # surf = ax.plot_surface(xx,yy,s)
            # plt.savefig("figure_1_%s_%d.png" % (names[j],i), type="png", dpi=300)
            # plt.close()
            # garbage.collect()

        # plt.legend(loc=0)
        # plt.title(names[j] + " Method - Relative Errors")
        # plt.xlabel("Iterations")
        # plt.ylabel("Relative Successive Error")
        # plt.savefig("figure_1_error_%s.png" % names[j], type="png", dpi=300)
        # plt.close()
        # garbage.collect()


if __name__ == "__main__":
    main()