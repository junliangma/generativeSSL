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
from models.bgssl import bgssl
from models.gssl import gssl
from models.VAE import VAE

def encode_onehot(labels):
    n, d = labels.shape[0], np.max(labels)+1
    return np.eye(d)[labels]

def load_mnist(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        train_set, _, test_set = cPickle.load(f)
    x_train, y_train = train_set[0], encode_onehot(train_set[1])
    x_test, y_test = test_set[0], encode_onehot(test_set[1])
    return x_train, y_train, x_test, y_test



### Script to run an experiment with generative SSL model ###

## argv[1] - dataset to use (mnist)
## argv[2] - proportion of training data labeled (300)
## argv[3] - Dataset seed (111)
## argv[4] - noise level in moons dataset / threshold for mnist reduction (0.1)


dataset, noise = sys.argv[1], sys.argv[4]
seed = int(sys.argv[3])

# Load data
target = './data/mnist.pkl.gz'
num_labeled, threshold = int(sys.argv[2]), float(noise)
labeled_batchsize, unlabeled_batchsize = 100,100
data = mnist(target, threshold=threshold)
data.create_semisupervised(num_labeled)


# Prepare data structure
data = SSL_DATA(data.x_unlabeled, data.y_unlabeled, x_test=data.x_test, y_test=data.y_test,
           x_labeled=data.x_labeled, y_labeled=data.y_labeled, dataset=dataset, seed=seed)

# Plot results
cl = plt.cm.tab10(np.linspace(0,1,10))
test_labs = np.argmax(data.data['y_test'], 1)

## Load and initialize models
# SSLAPD

z_dim = 50
learning_rate = (5e-4,)
initVar = -12.
architecture = [500, 500]
n_epochs = 10
temperature_epochs=None
start_temp = 0.0
type_px = 'Bernoulli'
binarize = True
batchnorm = False
logging = False


sslapd = bgssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp, BATCHNORM=batchnorm,
                LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, initVar=initVar, verbose=1, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, eval_samps=2000, logging=logging)
sslapd._data_init(data)
z_sslapd = sslapd.encode_new(data.data['x_test'].astype('float32'))

tf.reset_default_graph()





# SSLPE

z_dim = 50
learning_rate = (5e-4, )
architecture = [500, 500]
n_epochs = 10
type_px = 'Bernoulli'
temperature_epochs, start_temp = None, 0.0
l2_reg = 0.0
batchnorm = False
binarize, logging = True, False
verbose = 1

sslpe = gssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp, eval_samps=2000,
                l2_reg=l2_reg, BATCHNORM=batchnorm, LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, verbose=verbose, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)
sslpe._data_init(data)
z_sslpe = sslpe.encode_new(data.data['x_test'].astype('float32'))

tf.reset_default_graph()


## VAE

z_dim = 50
learning_rate = (1e-3,)
architecture = [500, 500]
n_epochs = 150
start_temp, temperature_epochs = 0.0, 1
batchsize=1024
type_px = 'Bernoulli'
batchnorm = False
binarize = True
logging = False


vae = VAE(LEARNING_RATE=learning_rate, Z_DIM=z_dim, NUM_HIDDEN=architecture, BATCH_SIZE=batchsize, start_temp=start_temp, ckpt='vae-mnist-50-15000',
            BATCHNORM=batchnorm, NUM_EPOCHS=n_epochs, temperature_epochs=temperature_epochs, TYPE_PX=type_px, BINARIZE=binarize)
vae._data_init(data)
z_vae = vae.encode_new(data.data['x_test'].astype('float32'))

# TSNE and plot
t1 = tsne(n_components=2, random_state=0)
t2 = tsne(n_components=2, random_state=0)
t3 = tsne(n_components=2, random_state=0)

print('Starting TSNE transform for SSLAPD encoding...')
sslapd_reduced = t1.fit_transform(z_sslapd)
print('Starting TSNE transform for SSLPE encoding...')
sslpe_reduced = t2.fit_transform(z_sslpe)
print('Starting TSNE transform for VAE encoding...')
vae_reduced = t3.fit_transform(z_vae)
print('Done with TSNE transformations...')


cl = plt.cm.tab10(np.linspace(0,1,10))
test_labs = np.argmax(data.data['y_test'], 1)
plt.figure(figsize=(8,10), frameon=False)
for digit in range(10):
    indices = np.where(test_labs==digit)[0]
    plt.scatter(sslapd_reduced[indices, 0], sslapd_reduced[indices, 1], c=cl[digit], label='Digit: '+str(digit))
plt.legend()
plt.savefig('mnist_samps/sslapd_encode', bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,10), frameon=False)
for digit in range(10):
    indices = np.where(test_labs==digit)[0]
    plt.scatter(sslpe_reduced[indices, 0], sslpe_reduced[indices, 1], c=cl[digit], label='Digit: '+str(digit))
plt.legend()
plt.savefig('mnist_samps/sslpe_encode', bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,10), frameon=False)
for digit in range(10):
    indices = np.where(test_labs==digit)[0]
    plt.scatter(vae_reduced[indices, 0], vae_reduced[indices, 1], c=cl[digit], label='Digit: '+str(digit))
plt.legend()
plt.savefig('mnist_samps/vae_encode', bbox_inches='tight')
plt.close()



