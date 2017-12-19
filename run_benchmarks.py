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
import data.benchmark_loader as loader
from models.m2 import m2
from models.adgm import adgm
from models.sdgm import sdgm
from models.blendedv2 import blended
from models.sblended import sblended
from models.b_blended import b_blended

### Script to run an MNIST experiment with generative SSL models ###

## argv[1] - Dataset to use (bci, coil2, coil, digit1, g241c, g241n, secstr, usps)
## argv[2] - model name (m2, blended, b_blended, sdgm, skip_blended)
## argv[3] - number of labeled instances to use

# Experiment parameters
dataset, modelName, numLabeled = sys.argv[1], sys.argv[2], int(sys.argv[3])
# Specify model parameters

if dataset=='g241c':
    l_bs, u_bs = 100, 512
    lr = (3e-4,)
    n_z, n_a = 10, 10
    n_hidden = [400, 400]
    n_epochs = 40
    x_dist = 'Gaussian'
    temp_epochs, start_temp = None, 0.0
    l2_reg, initVar, alpha = 1., -10., 1.1
    beta = 0.02
    batchnorm, mc_samps = True, 1
    eval_samps = None
    binarize, logging, verbose = False, False, 2

if dataset=='secstr':
    l_bs, u_bs = 100, 512
    lr = (3e-4,)
    n_z, n_a = 100, 100
    n_hidden = [400, 400]
    n_epochs = 40
    x_dist = 'Bernoulli'
    temp_epochs, start_temp = None, 0.0
    l2_reg, initVar, alpha = .1, -10., 1.1
    beta = 0.01
    batchnorm, mc_samps = True, 1
    eval_samps = None
    binarize, logging, verbose = False, False, 2



test_ll, test_ac = [],[]
for run in range(10):
    print("Starting work on run: {}".format(run))
    Data = loader.getBenchmarkSet(dataset, numLabeled, run)
    n_x, n_y = Data.INPUT_DIM, Data.NUM_CLASSES
    #pdb.set_trace()
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

test_ac, test_ll = np.array(test_ac), np.array(test_ll)
np.save('./output/benchmarks/'+dataset+'_'+modelName+'_mc'+str(mc_samps)+'_loglikelihood.npy', test_ll)
np.save('./output/benchmarks/'+dataset+'_'+modelName+'_mc'+str(mc_samps)+'_accuracy.npy', test_ac)



#np.save('./output/'+modelName+'_accuracy.npy', accuracy)
#np.save('./output/'+modelName+'_loglikelihood.npy', loglik)
#np.save('./output/beta.npy', betaset)
