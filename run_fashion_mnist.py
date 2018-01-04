import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
from sklearn.metrics import log_loss
from sklearn.manifold import TSNE as tsne
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from data.mnist import mnist
from models.m2 import m2
from models.adgm import adgm
from models.sdgm import sdgm
from models.blendedv2 import blended
from models.sblended import sblended
from models.b_blended import b_blended

### Script to run an MNIST experiment with generative SSL models ###

## argv[1] - number of labels from each class
## argv[2] - l2 regularization 
## argv[3] - noise level in moons dataset / Threshold for reduction in mnist
## argv[4] - model to use (m2, adgm, sdgm, sslpe, b_sdgm, msdgm)
## argv[5] - number of runs per beta
## argv[6] - number of MC samples

# Experiment parameters
num_labeled, threshold = int(sys.argv[1]), float(sys.argv[3])
seeds = [111,222,333,444,555,666,777,888,999]
modelName = sys.argv[4]
# Load and conver data to relevant type
target = './data/fashion_mnist.pkl.gz'
l_bs, u_bs = 100,100
betaset, runs = [0.01], int(sys.argv[5])
accuracy, loglik = np.zeros((runs, len(betaset))), np.zeros((runs,len(betaset)))
# Specify model parameters
lr = (3e-4,)
n_z, n_a = 100, 100
n_hidden = [500, 500]
n_epochs = 200
x_dist = 'Bernoulli'
temp_epochs, start_temp = None, 0.0
l2_reg, initVar, alpha = float(sys.argv[2]), -10., 1.1
batchnorm, mc_samps = True, int(sys.argv[6])
eval_samps = 1000
binarize, logging, verbose = True, False, 2

test_ll, test_ac = [],[]
for betaNum, beta in enumerate(betaset):
    print("beta = {:5.3f}".format(beta)) 
    for run in range(runs):
	seed = seeds[run]
	print("Starting work on run: {}".format(run))
        data = mnist(target, threshold=threshold)
        data.create_semisupervised(num_labeled)
        epoch2steps, epoch_decay = data.n_train/u_bs, 50
        Data = SSL_DATA(data.x_unlabeled, data.y_unlabeled, x_test=data.x_test, y_test=data.y_test, 
        	    x_labeled=data.x_labeled, y_labeled=data.y_labeled, dataset='mnist', seed=seed)
        n_x, n_y = Data.INPUT_DIM, Data.NUM_CLASSES 
        
        if modelName == 'm2':
            # standard M2: Kingma et al. (2014)
            model = m2(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
        
        elif modelName == 'adgm':
            # auxiliary DGM: Maaloe et al. (2016)
            model = adgm(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
        		    
        elif modelName == 'sdgm':
            # skip DGM: Maaloe et al. (2016)
            model = sdgm(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
        
        elif modelName == 'blended':
            # blending discriminative and generative models: unpublished
            model = blended(n_x, n_y, n_z, n_hidden, x_dist=x_dist, beta=beta, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
        
        elif modelName == 'sblended':
            # blending discriminative and generative models with skip deep stochastic layers: unpublished
            model = sblended(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, beta=beta, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
	
	elif modelName == 'b_blended':
            # blending Bayesian discriminative and regular generative models: unpublished
	    model = b_blended(n_x, n_y, n_z, n_hidden, initVar=initVar, x_dist=x_dist, beta=beta, alpha=alpha, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
        
        # Train model and measure performance on test set
        model.train(Data, n_epochs, l_bs, u_bs, lr, eval_samps=eval_samps, temp_epochs=temp_epochs, start_temp=start_temp, binarize=binarize, logging=logging, verbose=verbose)
        #train_curve.append(model.train_curve)
	#test_curve.append(model.test_curve) 
        preds_test = model.predict_new(Data.data['x_test'].astype('float32'))
        acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(Data.data['y_test'],1)), -log_loss(Data.data['y_test'], preds_test)
        print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))
	test_ll.append(ll)
	test_ac.append(acc)
	tf.reset_default_graph()

nl = num_labeled
test_ac, test_ll = np.array(test_ac), np.array(test_ll)
np.save('./output/fashion/'+modelName+'_nl'+str(nl)+'_mc'+str(mc_samps)+'_test_loglikelihood.npy', test_ll)
np.save('./output/fashion/'+modelName+'_nl'+str(nl)+'_mc'+str(mc_samps)+'_test_accuracy.npy', test_ac)



#np.save('./output/'+modelName+'_accuracy.npy', accuracy)
#np.save('./output/'+modelName+'_loglikelihood.npy', loglik)
#np.save('./output/beta.npy', betaset)
