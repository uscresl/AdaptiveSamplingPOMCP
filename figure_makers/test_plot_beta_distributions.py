import itertools
import os
os.chdir("..")

import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np

# less_than_one = list(np.linspace(0,1,5))
# print(less_than_one)
# greater = list(range(10))
# print(greater)
# p = less_than_one + greater
# print(p)
# params = list(itertools.product(p,p))
# print(params)
# print(len(params))
alpha_param = [.75,1,2,3,4,5,6]
beta_param =  [.75,1,2,3,4,5,6]


params = itertools.product(alpha_param,beta_param)

plt.figure()
for alpha_param,beta_param in params:
    xs = np.linspace(0,1,1000)
    ys = beta.cdf(xs,alpha_param,beta_param)
    plt.plot(xs,ys)
plt.title("Grid Search Curves")
plt.ylabel("Cumulative Rollout")
plt.xlabel("Fraction of trajectory")
plt.savefig("grid_search.pdf",dpi=300)
plt.show()

