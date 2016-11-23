from __future__ import division

import numpy as np

def get_mesh(power):
    N = 2**power+1
    x,h = np.linspace(0,1,N,retstep=True)
    x = x[1:len(x)-1:]
    X,Y = np.meshgrid(x,x)
    return X,Y,N,h