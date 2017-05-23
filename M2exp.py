import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.kingmaM2 import M2
import pdb

### Script to run an experiment with the model

## argv[1] - dataset to use (moons, digits)
dataset = sys.argv[1]

if dataset == 'moons':
    target = './data/moons.pkl'
    z_dim = 4
    learning_rate = 1e-3
    architecture = [10,10]
    n_epochs = 250
    type_px = 'Gaussian'

elif dataset == 'digits': 
    target = './data/digits.pkl'
    z_dim = 50
    learning_rate = 1e-3
    architecture = [200,200]
    n_epochs = 500
    type_px = 'Bernoulli'


labeled_batchsize, unlabeled_batchsize = 32,128
labeled_proportion = 0.5
with open(target, 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']
data = SSL_DATA(x,y, labeled_proportion=labeled_proportion) 
model = M2(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, 
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, 
		NUM_EPOCHS=n_epochs, TYPE_PX=type_px)
model.fit(data)
