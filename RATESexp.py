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
from models.gssl import gssl
from models.bgssl import bgssl
from models.kingmaM2 import M2 as m2


### Script to run an experiment with fixed data as in ADGM paper ###

## argv[1] - noise level to work with

n_epoch, temp_epoch = 150, 15 
noise = sys.argv[1]
target = './data/moons_semi_' + noise + '.pkl'
labeled_batchsize, unlabeled_batchsize = 6,100

with open(target, 'rb') as f:
    data = pickle.load(f)

tf.reset_default_graph()
x, y, xtest, ytest = data['x'], data['y'], data['x_test'], data['y_test']
x_l, y_l = data['x_labeled'], data['y_labeled'] 
data = SSL_DATA(x,y, x_test=xtest, y_test=ytest, x_labeled=x_l, y_labeled=y_l, dataset='moons_adgm') 

gssl_bn = gssl(Z_DIM=5, LEARNING_RATE=(3e-4,), NUM_HIDDEN=[128,128], ALPHA=0.1, BINARIZE=False, BATCHNORM=True, temperature_epochs=temp_epoch, start_temp=0.0, LABELED_BATCH_SIZE=6, 
	     UNLABELED_BATCH_SIZE=100, verbose=1, NUM_EPOCHS=n_epoch, TYPE_PX='Gaussian', logging=False)
gssl_bn.fit(data)
tf.reset_default_graph()

data = SSL_DATA(x,y, x_test=xtest, y_test=ytest, x_labeled=x_l, y_labeled=y_l, dataset='moons_adgm') 
gssl_no_bn = gssl(Z_DIM=5, LEARNING_RATE=(3e-4,750), NUM_HIDDEN=[128,128], ALPHA=0.1, BINARIZE=False, BATCHNORM=False, temperature_epochs=temp_epoch, start_temp=0.0, LABELED_BATCH_SIZE=6, 
	          UNLABELED_BATCH_SIZE=128, verbose=1, NUM_EPOCHS=n_epoch, TYPE_PX='Gaussian', logging=False)
gssl_no_bn.fit(data)
tf.reset_default_graph()

data = SSL_DATA(x,y, x_test=xtest, y_test=ytest, x_labeled=x_l, y_labeled=y_l, dataset='moons_adgm') 
bgssl_bn = bgssl(Z_DIM=5, LEARNING_RATE=(7e-4,), NUM_HIDDEN=[128,128], ALPHA=0.1, BINARIZE=False, BATCHNORM=True, LABELED_BATCH_SIZE=6, 
   	      UNLABELED_BATCH_SIZE=128, verbose=1, NUM_EPOCHS=n_epoch, temperature_epochs=temp_epoch, initVar=-5.5, TYPE_PX='Gaussian', logging=False)
bgssl_bn.fit(data)
tf.reset_default_graph()

data = SSL_DATA(x,y, x_test=xtest, y_test=ytest, x_labeled=x_l, y_labeled=y_l, dataset='moons_adgm') 
bgssl_no_bn = bgssl(Z_DIM=5, LEARNING_RATE=(3e-4,750), NUM_HIDDEN=[128,128], ALPHA=0.1, BINARIZE=False, BATCHNORM=False, LABELED_BATCH_SIZE=6, 
   	      UNLABELED_BATCH_SIZE=128, verbose=1, NUM_EPOCHS=n_epoch, temperature_epochs=temp_epoch, initVar=-5.5, TYPE_PX='Gaussian', logging=False)
bgssl_no_bn.fit(data)
    

plt.figure()
plt.plot(gssl_bn.epoch_test_acc, 'navy', label='SSLPE - BatchNorm', ls='-')
plt.plot(gssl_no_bn.epoch_test_acc, 'firebrick', label='SSLPE', ls='-')
plt.plot(bgssl_bn.epoch_test_acc, 'forestgreen', label='SSLAPD - BatchNorm', ls='-')
plt.plot(bgssl_no_bn.epoch_test_acc, 'darkmagenta', label='SSLAPD', ls='-')
plt.legend()
plt.savefig('./convergence/test_acc', bbox_inches='tight')


plt.figure()
plt.plot(gssl_bn.train_elbo, 'navy', label='SSLPE - BatchNorm', ls='-')
plt.plot(gssl_no_bn.train_elbo, 'firebrick', label='SSLPE', ls='-')
plt.plot(bgssl_bn.train_elbo, 'forestgreen', label='SSLAPD - BatchNorm', ls='-')
plt.plot(bgssl_no_bn.train_elbo, 'darkmagenta', label='SSLAPD', ls='-')
plt.ylim([0, 35])
plt.legend()
plt.savefig('./convergence/train_elbo', bbox_inches='tight')

results = {'g_bn_acc':gssl_bn.epoch_test_acc, 'g_bn_elbo':gssl_bn.train_elbo,
	   'g_no_bn_acc':gssl_no_bn.epoch_test_acc, 'g_no_bn_elbo':gssl_no_bn.train_elbo,
	   'b_bn_acc':bgssl_bn.epoch_test_acc, 'b_bn_elbo':bgssl_bn.train_elbo,
	   'b_no_bn_acc':bgssl_no_bn.epoch_test_acc, 'b_no_bn_elbo':bgssl_no_bn.train_elbo}


with open('./convergence/results.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
