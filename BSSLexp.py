import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import pickle, sys, pdb, gzip, cPickle
import numpy as np
from sklearn.metrics import log_loss
import tensorflow as tf
from data.SSL_DATA import SSL_DATA
from data.mnist import mnist
from models.bgssl import bgssl

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

## argv[1] - dataset to use (moons, digits, mnist)
## argv[2] - proportion of training data labeled
## argv[3] - Dataset seed
## argv[4] - noise level in moons dataset / threshold for mnist reduction


dataset, noise = sys.argv[1], sys.argv[4]
seed = int(sys.argv[3])


if dataset == 'moons':
    target = './data/moons_'+noise+'.pkl'
    labeled_proportion = float(sys.argv[2])
    labeled_batchsize, unlabeled_batchsize = 4,128
    
    z_dim = 5
    learning_rate = (2e-3,200)
    architecture = [128,128]
    n_epochs = 100
    temperature_epochs = 49 
    initVar = -10.
    start_temp = 0.0 
    type_px = 'Gaussian'
    verbose=1
    binarize = False
    logging = False

elif dataset == 'digits': 
    target = './data/digits.pkl'
    labeled_proportion = 0.2
    labeled_batchsize, unlabeled_batchsize = 6,32

    z_dim = 50
    learning_rate = 1e-3
    architecture = [200,200]
    n_epochs = 500
    type_px = 'Bernoulli'
    verbose = 1
    binarize = False
    logging = False

elif dataset == 'mnist':
    target = './data/mnist.pkl.gz'
    num_labeled, threshold = int(sys.argv[2]), float(noise)
    labeled_batchsize, unlabeled_batchsize = 100,100
    data = mnist(target, threshold=threshold)
    data.create_semisupervised(num_labeled)

    z_dim = 100
    learning_rate = (5e-4,)
    initVar = -12.
    architecture = [500, 500]
    n_epochs = 150
    temperature_epochs=None
    start_temp = 0.0 
    type_px = 'Bernoulli'
    binarize = True
    batchnorm = False
    logging = False

if dataset in ['moons', 'digits']:
    with open(target, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']
    data = SSL_DATA(x,y, labeled_proportion=labeled_proportion, dataset=dataset, seed=seed) 
elif dataset == 'mnist':
    data = SSL_DATA(data.x_unlabeled, data.y_unlabeled, x_test=data.x_test, y_test=data.y_test,
                    x_labeled=data.x_labeled, y_labeled=data.y_labeled, dataset=dataset, seed=seed)

model = bgssl(Z_DIM=z_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp, BATCHNORM=batchnorm,
		LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, initVar=initVar, verbose=1, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, eval_samps=2000, logging=logging)
model.fit(data)


if dataset == 'moons':
    X,Y = np.mgrid[-2.5:3.0:0.1, -2.5:3.0:0.1]
    xy = np.vstack((X.flatten(), Y.flatten())).T 
    predictions = model.predict_new(xy.astype('float32'))

   
    range_vals = np.arange(-2.5,3.0,.1)
    zi = np.zeros(X.shape)
    for i, row_val in enumerate(range_vals):
	for j, col_val in enumerate(range_vals):
	    idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
	    zi[i,j] = predictions[idx[0],0] * 100  
     
    plt.contourf(X, Y, zi,cmap=plt.cm.coolwarm)

    
    preds_test = model.predict_new(data.data['x_test'].astype('float32'))
    preds = np.argmax(preds_test, axis=1)
    x0, x1 = data.data['x_test'][np.where(preds==0)], data.data['x_test'][np.where(preds==1)]
    
    plt.scatter(x0[:,0], x0[:,1], color='g', s=1)    
    plt.scatter(x1[:,0], x1[:,1], color='m', s=1)    
    
    xl,yl = data.data['x_l'], data.data['y_l']
    plt.scatter(xl[:,0],xl[:,1], color='black')
    plt.savefig('../experiments/Moons/trial', bbox_inches='tight')

    plt.figure()
    x,y = model._sample_xy(int(1e4))
    y_bin = np.argmax(y, axis=1)
    x0, x1 = x[np.where(y_bin==0)], x[np.where(y_bin==1)]

    plt.scatter(x0[:,0],x0[:,1], s=1, color='r')
    plt.scatter(x1[:,0],x1[:,1], s=1, color='b')
    plt.savefig('../experiments/Moons/sample_trial', bbox_inches='tight')


if dataset=='mnist':
    #preds_test = model.predict_new(data.data['x_test'].astype('float32'))
    #acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test)
    #print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))

    if z_dim ==2:
        z_test = model.encode_new(data.data['x_test'].astype('float32'))
        cl = plt.cm.tab10(np.linspace(0,1,10))
        test_labs = np.argmax(data.data['y_test'], 1)
        plt.figure(figsize=(8,10), frameon=False)
        plt.axis('off')
        for digit in range(10):
            indices = np.where(test_labs==digit)[0]	
            plt.scatter(z_test[indices, 0], z_test[indices, 1], c=cl[digit], label='Digit: '+str(digit))
        plt.legend()
        plt.savefig('mnist_samps/sslapd_encode', bbox_inches='tight')

    # plot n_samps x n_samps grid of random samples
    if threshold==-1.0:
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
        plt.savefig('./mnist_samps/sslapd_'+str(num_labeled)+'_samps_black', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,10), frameon=False)
        plt.axis('off')
        plt.imshow(canvas2, origin="upper", cmap="gray")
        plt.tight_layout()
        plt.savefig('./mnist_samps/sslapd_'+str(num_labeled)+'_samps_white', bbox_inches='tight')
        plt.close()
