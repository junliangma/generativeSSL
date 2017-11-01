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
from models.msdgm import msdgm
from models.bsdgm import bsdgm

### Script to run an MNIST experiment with generative SSL models ###

## argv[1] - proportion of training data labeled (or for mnist, number of labels from each class)
## argv[2] - Dataset seed
## argv[3] - noise level in moons dataset / Threshold for reduction in mnist
## argv[4] - model to use (m2, adgm, sdgm, sslpe, b_sdgm, msdgm)

# Experiment parameters
num_labeled, threshold = int(sys.argv[1]), float(sys.argv[3])
seed = int(sys.argv[2])
modelName = sys.argv[4]
# Load and conver data to relevant type
target = './data/mnist.pkl.gz'
l_bs, u_bs = 100,100
data = mnist(target, threshold=threshold)
data.create_semisupervised(num_labeled)
epoch2steps, epoch_decay = data.n_train/u_bs, 50
data = SSL_DATA(data.x_unlabeled, data.y_unlabeled, x_test=data.x_test, y_test=data.y_test, 
	    x_labeled=data.x_labeled, y_labeled=data.y_labeled, dataset='mnist', seed=seed)
n_x, n_y = data.INPUT_DIM, data.NUM_CLASSES

# Specify model parameters
lr = (3e-4,)
n_z, n_a = 100, 100
n_hidden = [500, 500]
n_epochs = 200
x_dist = 'Bernoulli'
temp_epochs, start_temp = None, 0.0
l2_reg, initVar, alpha = .05, -8, 0.1
batchnorm, mc_samps = True, 5
eval_samps = 1000
binarize, logging, verbose = True, False, 1


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
    model = msdgm(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 'sslpe':
    # sslpe: Gordon and Hernandez-Lobato (2017)
    model = sslpe(n_x, n_y, n_z, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 's_sslpe':
    # skip sslpe: unpublished
    model = skip_sslpe(n_x, n_y, n_z, n_a, n_hidden, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)

elif modelName == 'b_sdgm':
    # SDGM with Bayesian additional classifier: unpublished
    model = bsdgm(n_x, n_y, n_z, n_a, n_hidden, initVar=initVar, x_dist=x_dist, batchnorm=batchnorm, mc_samples=mc_samps, l2_reg=l2_reg)
	

# Train model and measure performance on test set
model.train(data, n_epochs, l_bs, u_bs, lr, eval_samps=eval_samps, temp_epochs=temp_epochs, start_temp=start_temp, binarize=binarize, logging=logging, verbose=verbose)

preds_test = model.predict_new(data.data['x_test'].astype('float32'))
acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test)
print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))

""" Additional (optional) fancy generative model performance things
## t-sne visualization
cl = plt.cm.tab10(np.linspace(0,1,10))
test_labs = np.argmax(data.data['y_test'], 1)
z_test = model.encode_new(data.data['x_test'].astype('float32'))
np.save('./z_ssple', z_test)
np.save('./test_labs_sslpe', test_labs)
t = tsne(n_components=2, random_state=0)
print('Starting TSNE transform for latent encoding...')
sslpe_reduced = t.fit_transform(z_test)
print('Done with TSNE transformations...')
plt.figure(figsize=(8,10), frameon=False)
for digit in range(10):
    indices = np.where(test_labs==digit)[0]
    plt.scatter(sslpe_reduced[indices, 0], sslpe_reduced[indices, 1], c=cl[digit], label='Digit: '+str(digit))
plt.legend()
plt.savefig('mnist_samps/sslpe_encode', bbox_inches='tight')

# plot n_samps x n_samps grid of random samples
if threshold < 0.0:
    n_samps = 10
    samps, _ = model._sample_xy(n_samps**2)
    canvas1 = np.empty((28*n_samps, 28*n_samps))
    canvas2 = np.empty((28*n_samps, 28*n_samps))
    k=0
    for i in range(n_samps):
        for j in range(n_samps):
            canvas1[(n_samps-i-1)*28:(n_samps-i)*28, j*28:(j+1)*28] = samps[k].reshape(28,28)
            canvas2[(n_samps-i-1)*28:(n_samps-i)*28, j*28:(j+1)*28] = (1-samps[k]).reshape(28,28)
            k +=1

    plt.figure(figsize=(8,10), frameon=False)
    plt.axis('off')
    plt.imshow(canvas1, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.savefig('./mnist_samps/sslpe_'+str(num_labeled)+'_samps_black', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,10), frameon=False)
    plt.axis('off')
    plt.imshow(canvas2, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.savefig('./mnist_samps/sslpe_'+str(num_labeled)+'_samps_white', bbox_inches='tight')
    plt.close()
"""
