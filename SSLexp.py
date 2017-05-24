import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.generativeSSL import generativeSSL

def encode_onehot(labels):
    n, d = labels.shape[0], np.max(labels)+1
    return np.eye(d)[labels]

def load_mnist(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        train_set, _, test_set = cPickle.load(f)
    x_train, y_train = train_set[0], encode_onehot(train_set[1])
    x_test, y_test = test_set[0], encode_onehot(test_set[1])
    return x_train, y_train, x_test, y_test



### Script to run an experiment with the model

## argv[1] - dataset to use (moons, digits, mnist)
dataset = sys.argv[1]
labeled_batchsize, unlabeled_batchsize = 64,256
labeled_proportion = 0.1

if dataset == 'moons':
    target = './data/moons.pkl'
    z_dim = 4
    learning_rate = 1e-2
    architecture = [10,10]
    n_epochs = 20
    type_px = 'Gaussian'

elif dataset == 'digits': 
    target = './data/digits.pkl'
    z_dim = 50
    learning_rate = 1e-3
    architecture = [200,200]
    n_epochs = 500
    type_px = 'Bernoulli'

elif dataset == 'mnist':
    target = './data/mnist.pkl.gz'
    x_train, y_train, x_test, y_test = load_mnist(target)
    
    z_dim = 50
    learning_rate = 3e-4
    architecture = [600, 600]
    n_epochs = 500
    type_px = 'Bernoulli'
    

if dataset in ['moons', 'digits']:
    with open(target, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']
    data = SSL_DATA(x,y, labeled_proportion=labeled_proportion, dataset=dataset) 
elif dataset == 'mnist':
    data = SSL_DATA(x_train, y_train, x_test=x_test, y_test=y_test, labeled_proportion=labeled_proportion, dataset=dataset)

model = generativeSSL(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, 
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs, TYPE_PX=type_px)
model.fit(data)
