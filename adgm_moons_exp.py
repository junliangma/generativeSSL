import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from models.gssl import gssl
from models.igssl import igssl
from models.bgssl import bgssl
from models.DNN import DNN 



### Script to run an experiment with fixed data as in ADGM paper ###

## argv[1] - model to use (gssl, bgssl, m2, dnn)
## argv[2] - noise level to work with


model_type, noise = sys.argv[1], sys.argv[2]
target = './data/moons_semi_' + noise + '.pkl'
labeled_batchsize, unlabeled_batchsize = 6,100

if model_type == 'gssl':
    z_dim = 5
    learning_rate = (1e-3,300)
    architecture = [128,128]
    n_epochs = 150
    temperature_epochs = 75
    start_temp = 0.0
    type_px = 'Gaussian'
    binarize = False
    logging = False
    model = gssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp,
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)


if model_type == 'igssl':
    z_dim = 2
    learning_rate = 3e-3
    architecture = [100,100]
    n_epochs = 300
    type_px = 'Gaussian'
    binarize = False
    logging = False
    model = igssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize,
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)


if model_type == 'bgssl':
    z_dim = 5
    learning_rate = (6e-4, 300)
    architecture = [128,128]
    n_epochs = 500 
    temperature_epochs = 250 
    start_temp = 0.0
    initVar = -5.5
    type_px = 'Gaussian'
    binarize = False
    logging = False
    model = bgssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize,
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs, 
                temperature_epochs=temperature_epochs, initVar=initVar, TYPE_PX=type_px, logging=logging)


if model_type == 'dnn':
    learning_rate = 1e-2
    architecture = [128,128]
    batch_size = 6
    n_epochs = 200
    logging = False
    model = DNN(ARCHITECTURE=architecture, BATCH_SIZE=batch_size, NUM_EPOCHS=n_epochs, LEARNING_RATE=learning_rate, logging=logging)

with open(target, 'rb') as f:
    data = pickle.load(f)

x, y, xtest, ytest = data['x'], data['y'], data['x_test'], data['y_test']
x_l, y_l = data['x_labeled'], data['y_labeled'] 
data = SSL_DATA(x,y, x_test=xtest, y_test=ytest, x_labeled=x_l, y_labeled=y_l, dataset='moons_adgm') 
model.fit(data)




### Plotting Results
X,Y = np.mgrid[-.5:4.0:0.1, 0.0:3:0.1]
xy = np.vstack((X.flatten(), Y.flatten())).T

print('Starting plotting work')
predictions = model.predict_new(xy.astype('float32'))


range_x = np.arange(-.5,4.0,.1)
range_y = np.arange(.0,3.0,.1)
zi = np.zeros(X.shape)
for i, row_val in enumerate(range_x):
    for j, col_val in enumerate(range_y):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        zi[i,j] = predictions[idx[0],0] * 100

plt.contourf(X, Y, zi,cmap=plt.cm.coolwarm)
print('Done with heat map')

preds_test = model.predict_new(data.data['x_test'].astype('float32'))
preds = np.argmax(preds_test, axis=1)
x0, x1 = data.data['x_test'][np.where(preds==0)], data.data['x_test'][np.where(preds==1)]

plt.scatter(x0[:,0], x0[:,1], color='g', s=1)
plt.scatter(x1[:,0], x1[:,1], color='m', s=1)

xl,yl = data.data['x_l'], data.data['y_l']
plt.scatter(xl[:,0],xl[:,1], color='black')
plt.savefig('../experiments/Moons/adgm_trial', bbox_inches='tight')

print('Done with test data')

if model_type!='dnn':
    plt.figure()
    x,y = model._sample_xy(int(1e4))
    y_bin = np.argmax(y, axis=1)
    x0, x1 = x[np.where(y_bin==0)], x[np.where(y_bin==1)]
    
    plt.scatter(x0[:,0],x0[:,1], s=1, color='r')
    plt.scatter(x1[:,0],x1[:,1], s=1, color='b')
    plt.savefig('../experiments/Moons/adgm_sample_trial', bbox_inches='tight')
