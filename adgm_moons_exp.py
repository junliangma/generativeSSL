import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from models.generativeSSL import generativeSSL
from models.bgssl import bgssl
from models.DNN import DNN as dnn



### Script to run an experiment with fixed data as in ADGM paper ###

## argv[1] - model to use (gssl, bgssl, m2, dnn)

model_type = sys.argv[1]
target = './data/moons_semi.pkl'
labeled_batchsize, unlabeled_batchsize = 6,256

if model_type == 'gssl':
    z_dim = 5
    learning_rate = 1e-2
    architecture = [10,10]
    n_epochs = 150
    type_px = 'Gaussian'
    binarize = False
    logging = False
    model = generativeSSL(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize,
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)

with open(target, 'rb') as f:
    data = pickle.load(f)
x, y, xtest, ytest = data['x'], data['y'], data['x_test'], data['y_test']
x_l, y_l = data['x_labeled'], data['y_labeled'] 
data = SSL_DATA(x,y, x_test=xtest, y_test=ytest, x_labeled=x_l, y_labeled=y_l, dataset='moons_adgm') 

model.fit(data)


xl,yl = data.data['x_l'], data.data['y_l']
x1 = xl[np.where(yl[:,1]==1)]
x0 = xl[np.where(yl[:,0]==1)]

X,Y = np.mgrid[-2:2.5:0.1, -2:2.5:0.1]
xy = np.vstack((X.flatten(), Y.flatten())).T 
predictions = model.predict_new(xy.astype('float32'))


range_vals = np.arange(-2.0,2.5,.1)
zi = np.zeros(X.shape)
for i, row_val in enumerate(range_vals):
    for j, col_val in enumerate(range_vals):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        zi[i,j] = predictions[idx[0],0] * 100  
 
plt.contourf(X, Y, zi,cmap=plt.cm.coolwarm)

plt.scatter(x1[:,0],x1[:,1], color='white')
plt.scatter(x0[:,0],x0[:,1], color='black')

plt.savefig('../experiments/Moons/moons_adgm_'+model_type, bbox_inches='tight')






