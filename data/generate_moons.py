import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.datasets import make_moons
import pickle
import pdb 

## argv[1] - noise level

noise = float(sys.argv[1])
x, y = make_moons(int(1e4), noise=noise)
y = np.eye(2)[y]
data = {'x':x, 'y':y}
target = './data/moons.pkl'
with open(target, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

plt.figure()
x0 = x[np.where(y[:,0]==1)]
x1 = x[np.where(y[:,1]==1)]
plt.scatter(x0[:,0], x0[:,1], color='r', s=1)
plt.scatter(x1[:,0], x1[:,1], color='b', s=1)
plt.savefig('./data/moons_plot', bbox_inches='tight')


