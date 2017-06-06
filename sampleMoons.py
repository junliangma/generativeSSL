import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
import tensorflow as tf
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



### Script to run an experiment with generative SSL model ###

## argv[1] - dataset to use (moons, digits, mnist)
## argv[2] - proportion of training data labeled
## argv[3] - Dataset seed



dataset = sys.argv[1]
if len(sys.argv)==4:
    seed = int(sys.argv[3])
else:
    seed = None


if dataset == 'moons':
    target = './data/moons.pkl'
    labeled_proportion = float(sys.argv[2])
    labeled_batchsize, unlabeled_batchsize = 4,128
    
    z_dim = 10
    learning_rate = 3e-3
    architecture = [100,100]
    n_epochs = 100
    type_px = 'Gaussian'
    binarize = False
    logging = False

elif dataset == 'digits': 
    target = './data/digits.pkl'
    labeled_proportion = 0.2
    labeled_batchsize, unlabeled_batchsize = 6,32

    z_dim = 50
    learning_rate = 1e-3
    architecture = [200,200]
    n_epochs = 500
    type_px = 'Bernoulli'
    binarize = False
    logging = False

elif dataset == 'mnist':
    target = './data/mnist.pkl.gz'
    labeled_proportion = 0.015
    labeled_batchsize, unlabeled_batchsize = 64,128
    x_train, y_train, x_test, y_test = load_mnist(target)

    z_dim = 100
    learning_rate = 5e-5
    architecture = [600, 600]
    n_epochs = 500
    type_px = 'Bernoulli'
    binarize = True
    logging = False

if dataset in ['moons', 'digits']:
    with open(target, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']
    data = SSL_DATA(x,y, labeled_proportion=labeled_proportion, dataset=dataset, seed=seed) 
elif dataset == 'mnist':
    data = SSL_DATA(x_train, y_train, x_test=x_test, y_test=y_test, labeled_proportion=labeled_proportion, dataset=dataset, seed=seed)


model = generativeSSL(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize,
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=0, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)
model.fit(data)


if dataset == 'moons':
    x, y = model._sample_xy() 
     

    plt.scatter(x[:,0],x[:,1], color='white')
    
    plt.savefig('../experiments/Moons/gssl_sample', bbox_inches='tight')






