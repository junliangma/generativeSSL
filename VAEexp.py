import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, gzip, cPickle, pdb, sys
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
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
## argv[2] - Dataset seed
## argv[3] - noise level for data


dataset, noise = sys.argv[1], sys.argv[3]
seed = int(sys.argv[2])


if dataset == 'moons':
    target = './data/moons_' + noise + '.pkl'
    batchsize = 128
    z_dim = 5 
    learning_rate = (1e-3,)
    architecture = [128,128]
    n_epochs = 100
    temperature_epochs = 1 
    start_temp = 0.8
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
    data = SSL_DATA(x,y, dataset=dataset, seed=seed)
elif dataset == 'mnist':
    data = SSL_DATA(x_train, y_train, x_test=x_test, y_test=y_test, dataset=dataset, seed=seed)



model = VAE(LEARNING_RATE=learning_rate, Z_DIM=z_dim, NUM_HIDDEN=architecture, BATCH_SIZE=batchsize, start_temp=start_temp, 
	    NUM_EPOCHS=n_epochs, temperature_epochs=temperature_epochs, TYPE_PX=type_px, BINARIZE=binarize)
model.fit(data)


if dataset=='moons':
    f, (ax1, ax2) = plt.subplots(1, 2)
    x, xm, xs = model._generate_data(int(1e4))
    ax1.scatter(x[:,0], x[:,1], s=1, color='b')
    ax1.set_title('Samples')
    ax2.scatter(xm[:,0], xm[:,1], s=1, color='b')
    ax2.set_title('Means')
    plt.savefig('../experiments/Moons/vae_sample', bbox_inches='tight')

    x,y = data.data['x_test'], data.data['y_test']
    x0,x1 = x[np.where(y[:,0]==1)], x[np.where(y[:,1]==1)]
    z0, z1 = model.encode_new(x0.astype('float32')), model.encode_new(x1.astype('float32'))
    z0, z1 = z0[0], z1[0]

    plt.figure()
    plt.scatter(z0[:,0], z0[:,1], color='r', s=1)
    plt.scatter(z1[:,0], z1[:,1], color='b', s=1)
    plt.savefig('../experiments/Moons/vae_encode', bbox_inches='tight')




