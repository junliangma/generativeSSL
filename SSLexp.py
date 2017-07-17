import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from data.mnist import mnist
from models.gssl import gssl

### Script to run an experiment with generative SSL model ###

## argv[1] - dataset to use (moons, digits, mnist)
## argv[2] - proportion of training data labeled (or for mnist, number of labels from each class)
## argv[3] - Dataset seed
## argv[4] - noise level in moons dataset


dataset, noise = sys.argv[1], sys.argv[4]
seed = int(sys.argv[3])


if dataset == 'moons':
    target = './data/moons_'+noise+'.pkl'
    labeled_proportion = float(sys.argv[2])
    labeled_batchsize, unlabeled_batchsize = 4,128
    
    z_dim = 5
    learning_rate = (3e-4,300)
    architecture = [128,128]
    n_epochs = 100
    temperature_epochs = 99
    start_temp = 0.0  
    type_px = 'Gaussian'
    verbose=1
    binarize = False
    logging = False

elif dataset == 'digits': 
    target = './data/digits.pkl'
    labeled_proportion = float(sys.argv[2])
    labeled_batchsize, unlabeled_batchsize = 6,32

    z_dim = 50
    learning_rate = (1e-4, 300)
    architecture = [400,400]
    n_epochs = 500
    type_px = 'Bernoulli'
    verbose = 1
    binarize = False
    logging = False

elif dataset == 'mnist':
    target = './data/mnist.pkl.gz'
    num_labeled = int(sys.argv[2])
    labeled_batchsize, unlabeled_batchsize = 64,1024
    data = mnist(target)
    data.create_semisupervised(num_labeled)    

    z_dim = 50
    learning_rate = (5e-4,)
    architecture = [512, 512]
    n_epochs = 1500
    type_px = 'Bernoulli'
    temperature_epochs = 10
    start_temp = 0.0
    batchnorm = True
    binarize, logging = True, False
    verbose = 1

if dataset in ['moons', 'digits']:
    with open(target, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']
    data = SSL_DATA(x,y, labeled_proportion=labeled_proportion, dataset=dataset, seed=seed) 
elif dataset == 'mnist':
    data = SSL_DATA(data.x_unlabeled, data.y_unlabeled, x_test=data.x_test, y_test=data.y_test, 
		    x_labeled=data.x_labeled, y_labeled=data.y_labeled, dataset=dataset, seed=seed)

model = gssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp,
		BATCHNORM=batchnorm, LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=verbose, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)
model.fit(data)


if dataset == 'moons':
    X,Y = np.mgrid[-2.5:3.0:0.1, -2.5:3.0:0.1]
    xy = np.vstack((X.flatten(), Y.flatten())).T 
    predictions = model.predict_new(xy.astype('float32'))

   
    range_vals = np.arange(-2.5,3.0,.1)
    zi = np.zeros(X.shape)
    for i, row_val in enumerate(range_vals):
	for j, col_val in enumerate(range_vals):
	    idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
	    zi[i,j] = predictions[idx[0],0] * 100  
     
    plt.contourf(X, Y, zi,cmap=plt.cm.coolwarm)

    
    preds_test = model.predict_new(data.data['x_test'].astype('float32'))
    preds = np.argmax(preds_test, axis=1)
    x0, x1 = data.data['x_test'][np.where(preds==0)], data.data['x_test'][np.where(preds==1)]
    
    plt.scatter(x0[:,0], x0[:,1], color='g', s=1)    
    plt.scatter(x1[:,0], x1[:,1], color='m', s=1)    
    
    xl,yl = data.data['x_l'], data.data['y_l']
    plt.scatter(xl[:,0],xl[:,1], color='black', s=4)
    plt.savefig('../experiments/Moons/trial', bbox_inches='tight')

    plt.figure()
    x,y = model._sample_xy(int(1e4))
    y_bin = np.argmax(y, axis=1)
    x0, x1 = x[np.where(y_bin==0)], x[np.where(y_bin==1)]

    plt.scatter(x0[:,0],x0[:,1], s=1, color='r')
    plt.scatter(x1[:,0],x1[:,1], s=1, color='b')
    plt.savefig('../experiments/Moons/sample_trial', bbox_inches='tight')

    


