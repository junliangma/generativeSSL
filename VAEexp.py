import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, gzip, cPickle, pdb, sys
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.VAE import VAE

def encode_onehot(labels):
    n, d = labels.shape[0], np.max(labels)+1
    return np.eye(d)[labels]

def load_mnist(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
	train_set, _, test_set = cPickle.load(f)
    x_train, y_train = train_set[0], encode_onehot(train_set[1])
    x_test, y_test = test_set[0], encode_onehot(test_set[1])
    return SSL_DATA(x_train, y_train, x_test=x_test, y_test=y_test, dataset='MNIST')


### Script to run an experiment with standard VAE ###

## argv[1] - dataset to use (moons, digits, mnist)
## argv[2] - proportion of training data labeled
## argv[3] - Dataset seed


dataset = sys.argv[1]
labeled_proportion = float(sys.argv[2])
if len(sys.argv)==4:
    seed = int(sys.argv[3])
else:
    seed = None


if dataset == 'moons':
    target = './data/moons.pkl'
    batchsize = 128

    z_dim = 10
    learning_rate = 1e-3
    architecture = [100,100]
    n_epochs = 250
    type_px = 'Gaussian'
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

model = VAE(LEARNING_RATE=learning_rate, Z_DIM=z_dim, NUM_HIDDEN=architecture, BATCH_SIZE=batchsize, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, BINARIZE=binarize)
model.fit(data)


if dataset=='moons':
    x = model._generate_data(int(1e4))
    plt.scatter(x[:,0], x[:,1], s=1, color='gray')
    plt.savefig('../experiments/Moons/vae_sample', bbox_inches='tight')



