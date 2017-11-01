import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
from sklearn.metrics import log_loss
from sklearn.manifold import TSNE as tsne
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from data import half_moon_loader
from models.m2 import m2
from models.adgm import adgm
from models.sdgm import sdgm
from models.msdgm import msdgm
from models.bsdgm import bsdgm

### Script to run a toy moons experiment with generative SSL models ###

## argv[1] - noise level in moons dataset (0.1, 0.2)
## argv[2] - Dataset seed
## argv[3] - model to use (m2, adgm, sdgm, sslpe, bsdgm, msdgm)

# Experiment parameters
num_labeled, seed = 3, int(sys.argv[2])
modelName = sys.argv[3]

# load data (centered)
target = './data/moons_semi_'+sys.argv[1]+'.pkl'
with open(target, 'rb') as handle:
    moons = pickle.load(handle)
data = SSL_DATA(moons['x'], moons['y'], x_test=moons['x_test'], y_test=moons['y_test'], 
	    x_labeled=moons['x_labeled'], y_labeled=moons['y_labeled'], dataset='moons', seed=seed)

# load data (uncentered)
#moons = half_moon_loader.load_semi_supervised()
#data = SSL_DATA(moons[0][0], moons[0][1], x_test=moons[2][0], y_test=moons[2][1], 
#	    x_labeled=moons[1][0], y_labeled=moons[1][1], dataset='moons', seed=seed)

# data specifics
l_bs, u_bs = 6, 100
n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES


# Specify model parameters
lr = (3e-4,)
n_z, n_a = 10, 10
n_hidden = [100, 100]
n_epochs = 15
x_dist = 'Gaussian'
temp_epochs, start_temp = None, 0.0
l2_reg, initVar = 1., -8.0	
batchnorm, mc_samps = True, 10
alpha, eval_samps = 2., None
binarize, logging, verbose = False, False, 1


if modelName == 'm2':
    # standard M2: Kingma et al. (2014)
    model = m2(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 'adgm':
    # auxiliary DGM: Maaloe et al. (2016)
    model = adgm(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
		    
elif modelName == 'sdgm':
    # skip DGM: Maaloe et al. (2016)
    model = sdgm(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 'msdgm':
    # modified sdgm (copied classifier p(y|x,a)
    model = msdgm(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 'sslpe':
    # sslpe: Gordon and Hernandez-Lobato (2017)
    model = sslpe(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 's_sslpe':
    # skip sslpe: unpublished
    model = skip_sslpe(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 'bsdgm':
    # SDGM with Bayesian additional classifier: unpublished
    model = bsdgm(n_x, n_y, n_z, n_a, n_hidden, initVar=initVar, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
	

# Train model and measure performance on test set
model.train(data, n_epochs, l_bs, u_bs, lr, eval_samps=eval_samps, temp_epochs=temp_epochs, start_temp=start_temp, binarize=binarize, logging=logging, verbose=verbose)

#sys.exit()
preds_test = model.predict_new(data.data['x_test'].astype('float32'))
acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test)
print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))

if modelName=='b_sdgm':
    preds_test_q = model.predict_new_q(data.data['x_test'].astype('float32'))
    acc, ll = np.mean(np.argmax(preds_test_q,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test_q)
    print('Test Accuracy (q): {:5.3f}, Test log-likelihood (q): {:5.3f}'.format(acc, ll))


## Visualize predictions
range_x = np.arange(-2.,3.,.1)
range_y = np.arange(-1.5,2.,.1)
X,Y = np.mgrid[-2.:3.:.1, -1.5:2.:.1]
xy = np.vstack((X.flatten(), Y.flatten())).T

print('Starting plotting work')	
predictions = model.predict_new(xy.astype('float32'))

zi = np.zeros(X.shape)
for i, row_val in enumerate(range_x):
    for j, col_val in enumerate(range_y):
        idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
        zi[i,j] = predictions[idx[0],0] * 100

plt.figure()
plt.contourf(X,Y,zi,cmap=plt.cm.coolwarm)
print('Done with heat map')

preds_test = model.predict_new(data.data['x_test'].astype('float32'))
ll_test = -log_loss(data.data['y_test'], preds_test)
print('Final Test Log Likelihood: {:5.3f}'.format(ll_test))
 
preds = np.argmax(preds_test, axis=1)
x0, x1 = data.data['x_test'][np.where(preds==0)], data.data['x_test'][np.where(preds==1)]

plt.scatter(x0[:,0], x0[:,1], color='g', s=1)
plt.scatter(x1[:,0], x1[:,1], color='m', s=1)

xl,yl = data.data['x_l'], data.data['y_l']
plt.scatter(xl[:,0],xl[:,1], color='black')
plt.show()

if modelName == 'msdgm':
    predictions = model.predict_q_new(xy.astype('float32'))
    
    zi = np.zeros(X.shape)
    for i, row_val in enumerate(range_x):
        for j, col_val in enumerate(range_y):
            idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
            zi[i,j] = predictions[idx[0],0] * 100
    
    plt.figure()
    plt.contourf(X,Y,zi,cmap=plt.cm.coolwarm)
    print('Done with heat map')
    
    preds_test = model.predict_q_new(data.data['x_test'].astype('float32'))
    ll_test = -log_loss(data.data['y_test'], preds_test)
    print('Final Test Log Likelihood: {:5.3f}'.format(ll_test))
     
    preds = np.argmax(preds_test, axis=1)
    x0, x1 = data.data['x_test'][np.where(preds==0)], data.data['x_test'][np.where(preds==1)]
    
    plt.scatter(x0[:,0], x0[:,1], color='g', s=1)
    plt.scatter(x1[:,0], x1[:,1], color='m', s=1)
    
    xl,yl = data.data['x_l'], data.data['y_l']
    plt.scatter(xl[:,0],xl[:,1], color='black')
    plt.show()
