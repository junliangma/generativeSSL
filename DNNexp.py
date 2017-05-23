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

## argv[1] - dataset
dataset = sys.argv[1]

if dataset == 'moons':
    target = './data/moons.pkl'
    network = [15, 15]
    batchsize = 32
    n_epochs = 25
    learning_rate = 1e-3

elif dataset == 'digits':
    target = './data/digits.pkl'
    network = [150, 150]
    batchsize = 16
    n_epochs = 50
    learning_rate = 1e-3
 
with open(target, 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']

### subsample the data
#data_size = int(sys.argv[1])
#indices = np.arange(x.shape[0])
#np.random.shuffle(indices)
#x, y = x[indices[:data_size]], y[indices[:data_size]]

Data = SSL_DATA(x,y) 
model = DNN(ARCHITECTURE=network, BATCH_SIZE=batchsize, NUM_EPOCHS=n_epochs, LEARNING_RATE=learning_rate)
model.fit(Data)
