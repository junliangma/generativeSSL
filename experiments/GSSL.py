import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.generativeSSL import generativeSSL
import pdb

target = './data/moons.pkl'
with open(target, 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']
Moons = SSL_DATA(x,y, labeled_proportion=0.15) 
model = generativeSSL(Z_DIM=4, LEARNING_RATE=6e-3, NUM_HIDDEN=10, ALPHA=0.1, LABELED_BATCH_SIZE=4, UNLABELED_BATCH_SIZE=128, verbose=1, NUM_EPOCHS=100)
model.fit(Moons)
