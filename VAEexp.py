import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, gzip, cPickle, pdb, sys
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from scipy.stats import norm
from data.SSL_DATA import SSL_DATA
from data.mnist import mnist
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
    batchsize = 4096 
    z_dim = 5  
    learning_rate = (3e-3, 500)
    architecture = [128, 128]
    n_epochs = 1000
    temperature_epochs = 150 
    start_temp = 0.0
    type_px = 'Gaussian'
    binarize = False
    logging = False


elif dataset == 'mnist':
    target = './data/mnist.pkl.gz'
    data = mnist(target, threshold=-1.0)
    z_dim = 2 
    learning_rate = (1e-3,)
    architecture = [500, 500]
    n_epochs = 150
    start_temp, temperature_epochs = 0.0, 10
    batchsize=1024
    type_px = 'Bernoulli'
    batchnorm = False
    binarize = True
    logging = False


if dataset in ['moons', 'digits']:
    with open(target, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']
    data = SSL_DATA(x,y, dataset=dataset, seed=seed)
elif dataset == 'mnist':
    data = SSL_DATA(data.x_train, data.y_train, x_test=data.x_test, y_test=data.y_test, dataset=dataset, seed=seed)



model = VAE(LEARNING_RATE=learning_rate, Z_DIM=z_dim, NUM_HIDDEN=architecture, BATCH_SIZE=batchsize, start_temp=start_temp, 
	    BATCHNORM=batchnorm, NUM_EPOCHS=n_epochs, temperature_epochs=temperature_epochs, TYPE_PX=type_px, BINARIZE=binarize)
model.fit(data)


if dataset=='moons':
    #f, (ax1, ax2) = plt.subplots(1, 2)
    x, xm, xs = model._generate_data(int(16384))
    #ax1.scatter(x[:,0], x[:,1], s=1, color='b')
    #ax1.set_title('Samples')
    #ax2.scatter(xm[:,0], xm[:,1], s=1, color='b')
    #ax2.set_title('Means')
    plt.scatter(x[:,0], x[:,1], s=1, color='b')
    plt.title('Samples')
    plt.savefig('../experiments/Moons/vae_samples', bbox_inches='tight')

    plt.figure()
    plt.scatter(xm[:,0], xm[:,1], s=1, color='r')
    plt.title('Means')
    plt.savefig('../experiments/Moons/vae_mean', bbox_inches='tight')
    
    x,y = data.data['x_test'], data.data['y_test']
    x0,x1 = x[np.where(y[:,0]==1)], x[np.where(y[:,1]==1)]
    z0, z1 = model.encode_new(x0.astype('float32')), model.encode_new(x1.astype('float32'))
    z0, z1 = z0[0], z1[0]

    plt.figure()
    plt.scatter(z0[:,0], z0[:,1], color='r', s=1)
    plt.scatter(z1[:,0], z1[:,1], color='b', s=1)
    plt.savefig('../experiments/Moons/vae_encode', bbox_inches='tight')



if dataset=='mnist':
    # plot n_samps x n_samps grid of random samples 
    n_samps = 10
    samps = model._generate_data(n_samps^2)
    canvas = np.empty((28*n_samps, 28*n_samps))
    k = 0
    for i in range(n_samps):
        for j in range(n_samps):
            canvas[(n_samps-i-1)*28:(n_samps-i)*28, j*28:(j+1)*28] = samps[k].reshape(28,28)
            k+=1
    plt.figure(figsize=(8, 10), frameon=False)
    plt.axis('off')
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.savefig('./mnist_samps/samples', bbox_inches='tight')
    plt.close()
   
    # Draw latent manifold (only if dim(z) = 2 )
    if z_dim == 2:     
        nx = ny = 20
        x_values = np.linspace(.05, .95, nx)
        y_values = np.linspace(.05, .95, ny)

        canvas = np.empty((28*ny, 28*nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = np.array([[norm.ppf(xi), norm.ppf(yi)]]).astype('float32')
                x_mean = model.decode_new(z_mu)
                canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

        plt.figure(figsize=(8, 10), frameon=False)
        plt.axis('off')
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()
        plt.savefig('./mnist_samps/grid', bbox_inches='tight')
        plt.close()


