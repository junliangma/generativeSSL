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
from models.gssl import gssl



### Active learning experimentation on the moons data ###

## argv[1] - noise level to work with
## argv[2] - model checkpoint directory
## argv[3] - initial labeled proportion
## argv[4] - model to work with (b/gssl)
## argv[5] - acquisition function (predictive_entropy, bald (bgssl only), random (baseline))
## argv[6] - random seed

noise, ckpt, labeled_proportion, model_type, acq_func, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])

## Data definitions
target = './data/moons_semi_' + noise + '.pkl'
with open(target, 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']
data = SSL_DATA(x,y, labeled_proportion=labeled_proportion, dataset='moons', seed=seed)
num_labeled = data.data['x_l'].shape[0]


if model_type=='bgssl':
    z_dim = 5
    learning_rate = (7e-4,)
    architecture = [128,128]
    n_epochs = 50
    temperature_epochs = 10
    start_temp = 0.0
    initVar = -5.5
    type_px = 'Gaussian'
    batchnorm = True
    binarize, logging = False, False

elif model_type=='gssl':
    z_dim = 5
    learning_rate = (5e-4,)
    architecture = [128,128]
    n_epochs = 50
    temperature_epochs = 10 
    start_temp = 0.0
    type_px = 'Gaussian'
    batchnorm = True
    binarize, logging = False, False

acc, ll, iteration = [], [], 0
while num_labeled <= 30:
    labeled_batchsize, unlabeled_batchsize = num_labeled, 100
    
    if model_type=='bgssl':
        model = bgssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize, BATCHNORM=batchnorm,
                LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs,
                temperature_epochs=temperature_epochs, initVar=initVar, TYPE_PX=type_px, logging=logging, ckpt=ckpt)

    elif model_type=='gssl':
        model = gssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize, BATCHNORM=batchnorm, 
		temperature_epochs=temperature_epochs, start_temp=start_temp, LABELED_BATCH_SIZE=labeled_batchsize, 
		UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=1, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging, ckpt=ckpt)
    model.fit(data)
    _, idx_to_label = model._acquisition_new(data.data['x_u'].astype('float32'), acq_func)
    data.query(idx_to_label)
    
    preds = model.predict_new(data.data['x_test'].astype('float32'))
    ll.append(-log_loss(data.data['y_test'], preds))
    preds = np.argmax(preds, axis=1)
    acc.append(np.mean(preds==np.argmax(data.data['y_test'],axis=1)))
    iteration += 1
    print('Iteration: {}, Labeled: {}, Accuracy: {:5.3f}'.format(iteration, num_labeled, acc[-1]))
    num_labeled = data.data['x_l'].shape[0]
    data.reset_counters()
    del model
    tf.reset_default_graph()

np.savetxt('./al_results/acc.txt', np.array(acc), fmt='%.4f', newline=" ")
np.savetxt('./al_results/ll.txt', np.array(ll), fmt='%.4f', newline=" ")
