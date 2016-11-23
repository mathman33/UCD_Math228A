from __future__ import division

import numpy as np
from math import log

def restriction(u):
    power = int(log(u.shape[0]+1,2))-1
    coarse_grid = 2**power-1
    v = np.zeros((coarse_grid,coarse_grid))
    for i in xrange(0,coarse_grid):
        for j in xrange(0,coarse_grid):
            row_1 = u.item(((2*i),(2*j))) + 2*u.item(((2*i),(2*j)+1)) + u.item(((2*i),(2*j)+2))
            row_2 = 2*(u.item(((2*i)+1,(2*j))) + 2*u.item(((2*i)+1,(2*j)+1)) + u.item(((2*i)+1,(2*j)+2)))
            row_3 = u.item(((2*i)+2,(2*j))) + 2*u.item(((2*i)+2,(2*j)+1)) + u.item(((2*i)+2,(2*j)+2))
            v[i][j] = (1/16)*(row_1 + row_2 + row_3)
    return v

def interpolation(v,power):
    coarse_grid = 2**(power-1)-1
    fine_grid = 2**power-1
    u = np.zeros((fine_grid,fine_grid))
    for i in xrange(coarse_grid):
        for j in xrange(coarse_grid):
            # Top row
            u[(2*i)][(2*j)] += (1/4)*v[i][j]
            u[(2*i)][(2*j)+1] += (1/2)*v[i][j]
            u[(2*i)][(2*j)+2] += (1/4)*v[i][j]

            # Middle row
            u[(2*i)+1][(2*j)] += (1/2)*v[i][j]
            u[(2*i)+1][(2*j)+1] += v[i][j]
            u[(2*i)+1][(2*j)+2] += (1/2)*v[i][j]

            # Bottom row
            u[(2*i)+2][(2*j)] += (1/4)*v[i][j]
            u[(2*i)+2][(2*j)+1] += (1/2)*v[i][j]
            u[(2*i)+2][(2*j)+2] += (1/4)*v[i][j]
    return u

