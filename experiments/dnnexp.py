import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.DNN import DNN
import pdb

target = './data/moons.pkl'
with open(target, 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']
Moons = SSL_DATA(x,y) 
model = DNN(ARCHITECTURE=[4,4], BATCH_SIZE=128)
model.fit(Moons)
