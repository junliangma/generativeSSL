import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import sys, pdb, os, pickle, gzip, cPickle
import numpy as np
from data.SSL_DATA import SSL_DATA
from data.mnist import mnist
from models.DNN import DNN

### Run experiments with standard DNN ###

## argv[1] - dataset to use (moons, digits, mnist)
## argv[2] - proportion of training data labeled (number of labels for MNIST)
## argv[3] - dataset seed
## argv[4] - noise level to use (moons only)

dataset = sys.argv[1]
labeled_proportion = float(sys.argv[2])
seed = int(sys.argv[3])

if dataset == 'moons':
    noise = sys.argv[4]
    target = './data/moons_'+noise+'.pkl'
    network = [100, 100]
    batchsize = 4
    n_epochs = 50
    learning_rate = 1e-2

elif dataset == 'digits':
    target = './data/digits.pkl'
    network = [150, 150]
    batchsize = 16
    n_epochs = 50
    learning_rate = 1e-3
 
elif dataset == 'mnist':
    target = './data/mnist.pkl.gz'
    num_labeled = int(sys.argv[2])
    data = mnist(target)
    data.create_semisupervised(num_labeled)
    batchsize = 512
    network = [500, 500]
    n_epochs = 500
    learning_rate = 1e-3


if dataset in ['moons', 'digits']:
    with open(target, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']
    Data = SSL_DATA(x,y, labeled_proportion=labeled_proportion, seed=seed)
elif dataset == 'mnist':
    Data = SSL_DATA(data.x_unlabeled, data.y_unlabeled, x_labeled=data.x_labeled, y_labeled=data.y_labeled, 
		    x_test=data.x_test, y_test=data.y_test, dataset=dataset, seed=seed)


model = DNN(ARCHITECTURE=network, BATCH_SIZE=batchsize, NUM_EPOCHS=n_epochs, LEARNING_RATE=learning_rate)
model.fit(Data)


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


    preds_test = model.predict_new(Data.data['x_test'].astype('float32'))
    preds = np.argmax(preds_test, axis=1)
    x0, x1 = Data.data['x_test'][np.where(preds==0)], Data.data['x_test'][np.where(preds==1)]

    plt.scatter(x0[:,0], x0[:,1], color='g', s=1)
    plt.scatter(x1[:,0], x1[:,1], color='m', s=1)

    xl,yl = Data.data['x_l'], Data.data['y_l']
    plt.scatter(xl[:,0],xl[:,1], color='black')
    plt.savefig('../experiments/Moons/dnn_trial', bbox_inches='tight')


if dataset == 'mnist':
   preds_test = model.predict_new(data.data['x_test'].astype('float32'))
   acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test)
   print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))

