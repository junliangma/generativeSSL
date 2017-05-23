import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, gzip, cPickle
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.VAE import VAE
import pdb

def encode_onehot(labels):
    n, d = labels.shape[0], np.max(labels)+1
    return np.eye(d)[labels]

def load_mnist(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
	train_set, _, test_set = cPickle.load(f)
    x_train, y_train = train_set[0], encode_onehot(train_set[1])
    x_test, y_test = test_set[0], encode_onehot(test_set[1])
    return SSL_DATA(x_train, y_train, x_test=x_test, y_test=y_test, dataset='MNIST')


MNIST = load_mnist()
model = VAE(LEARNING_RATE=1e-5, Z_DIM=50, NUM_HIDDEN=[400, 400], BATCH_SIZE=128, NUM_EPOCHS=25)
model.fit(MNIST)
