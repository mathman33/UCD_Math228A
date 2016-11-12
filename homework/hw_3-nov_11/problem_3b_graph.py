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
    # with open("sor_data_3.p", "r") as handle:
    #     sor_data = pickle.load(handle)
    with open("direct_data_3b.p", "r") as handle:
        direct_data = pickle.load(handle)

    hs = [0.015625, 0.0078125]
    labels = [r"$2^{-6}$",r"$2^{-7}$"]

    # plt.figure()
    # for i in xrange(4):
    #     t = sor_data["times"][i]
    #     s = sor_data["solns"][i]
    #     r = sor_data["rel_ers"][i]
    #     h = hs[i]

    #     print t

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

    print direct_data["times"]

    # plt.legend(loc=0)
    # plt.title("SOR Method - Relative Errors")
    # plt.xlabel("Iterations")
    # plt.ylabel("Relative Successive Error")
    # plt.savefig("figure_3_SOR_error_SOR.png", type="png", dpi=300)
    # plt.close()
    # garbage.collect()


if __name__ == "__main__":
    main()