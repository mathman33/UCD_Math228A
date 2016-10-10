from __future__ import division
import numpy as np
from math import exp
import matplotlib.pyplot as plt

x = 2
true_val = exp(2)

def approximation(h):
    return (8/(3*(h**2)))*exp(x - (h/2)) - (4/(h**2))*exp(x) + (4/(3*(h**2)))*exp(x + h)

h = np.linspace(1, 0.01, 100)
approx_vals = [approximation(H) for H in h]
errors = [abs(i - true_val) for i in approx_vals]

plt.figure()
plt.plot(h, errors, "--", label="absolute error")
plt.plot(h, (1/6)*h*exp(x), label=r"$\frac{1}{6}e^2h$")
plt.legend(loc=0)
plt.xlabel(r"$h$")
plt.ylabel("absolute error")
plt.savefig("problem_3.png", dpi=300)
plt.close()


