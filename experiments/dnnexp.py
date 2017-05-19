import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle
import sys
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.DNN import DNN
import pdb

## argv[1] - data size


target = './data/moons.pkl'
with open(target, 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']

### subsample the data
data_size = int(sys.argv[1])
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x, y = x[indices[:data_size]], y[indices[:data_size]]
Moons = SSL_DATA(x,y) 
model = DNN(ARCHITECTURE=[100], BATCH_SIZE=16, NUM_EPOCHS=25)
model.fit(Moons)
