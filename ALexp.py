import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
from sklearn.metrics import log_loss
import pickle, sys, pdb, gzip, cPickle
import numpy as np
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from models.bgssl import bgssl




### Initial experimentation with the acquisition functions ###

## argv[1] - noise level to work with
## argv[2] - model checkpoint directory
## argv[3] - train (yes) or load (no) model

noise, ckpt, train = sys.argv[1], sys.argv[2], sys.argv[3]
target = './data/moons_semi_' + noise + '.pkl'
labeled_batchsize, unlabeled_batchsize = 6,100

z_dim = 5
learning_rate = (3e-4,1000)
architecture = [128,128]
n_epochs = 100
temperature_epochs = 99
start_temp = 0.0
initVar = -5.5
type_px = 'Gaussian'
binarize = False
logging = False
model = bgssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize,
            LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs,
            temperature_epochs=temperature_epochs, initVar=initVar, TYPE_PX=type_px, logging=logging, ckpt=ckpt)



with open(target, 'rb') as f:
    data = pickle.load(f)

x, y, xtest, ytest = data['x'], data['y'], data['x_test'], data['y_test']
x_l, y_l = data['x_labeled'], data['y_labeled']
data = SSL_DATA(x,y, x_test=xtest, y_test=ytest, x_labeled=x_l, y_labeled=y_l, dataset='moons_adgm')

if train=='yes':
    model.fit(data)
else:
    model._data_init(data)


#### Plotting ####

range_x = np.arange(-2.5,3.5,.1)
range_y = np.arange(-2.,2.5,.1)
X,Y = np.mgrid[-2.5:3.5:.1, -2.:2.5:.1]
xy = np.vstack((X.flatten(), Y.flatten())).T

print('Starting plotting work')
pred_ent, _ = model._acquisition_new(xy.astype('float32'), 'predictive_entropy')
bald, _ = model._acquisition_new(xy.astype('float32'), 'bald')
var_ratios, _ = model._acquisition_new(xy.astype('float32'), 'var_ratios')


z1, z2, z3 = np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)
for i, row_val in enumerate(range_x):
    for j, col_val in enumerate(range_y):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        z1[i,j] = pred_ent[idx[0]]
        z2[i,j] = bald[idx[0]]
        z3[i,j] = var_ratios[idx[0]]

pool = data.data['x_train']

ent_pool, ent_max = model._acquisition_new(pool.astype('float32'), 'predictive_entropy')
bald_pool, bald_max = model._acquisition_new(pool.astype('float32'), 'bald')
var_ratios_pool, var_ratios_max = model._acquisition_new(pool.astype('float32'), 'var_ratios')

plt.contourf(X, Y, z1,cmap=plt.cm.coolwarm)
plt.scatter(data.data['x_train'][:,0],data.data['x_train'][:,1], color='gray', alpha=0.1)
plt.savefig('../experiments/Moons/predictive_entropy', bbox_inches='tight')

plt.figure()
plt.contourf(X, Y, z2,cmap=plt.cm.coolwarm)
plt.scatter(data.data['x_train'][:,0],data.data['x_train'][:,1], color='gray', alpha=0.1)
plt.savefig('../experiments/Moons/bald', bbox_inches='tight')

plt.figure()
plt.contourf(X, Y, z3,cmap=plt.cm.coolwarm)
plt.scatter(data.data['x_train'][:,0],data.data['x_train'][:,1], color='gray', alpha=0.1)
plt.savefig('../experiments/Moons/variational_ratios', bbox_inches='tight')

plt.figure()
plt.scatter(data.data['x_train'][:,0],data.data['x_train'][:,1], color='gray', alpha=0.1)
plt.scatter(pool[ent_max][0], pool[ent_max][1], color='red', label='Predictive Entropy')
plt.scatter(pool[bald_max][0], pool[bald_max][1], color='green', label='BALD')
plt.scatter(pool[var_ratios_max][0], pool[var_ratios_max][1], color='blue', label='Variation Ratios')
plt.legend()
plt.savefig('../experiments/Moons/selection', bbox_inches='tight')








