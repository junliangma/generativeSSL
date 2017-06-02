import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import rc
import sys, pdb, os, pickle
import numpy as np
from data.SSL_DATA import SSL_DATA
from models.DNN import DNN

def encode_onehot(labels):
    n, d = labels.shape[0], np.max(labels)+1
    return np.eye(d)[labels]

def load_mnist(path='data/mnist.pkl.gz'):
    with gzip.open(path, 'rb') as f:
        train_set, _, test_set = cPickle.load(f)
    x_train, y_train = train_set[0], encode_onehot(train_set[1])
    x_test, y_test = test_set[0], encode_onehot(test_set[1])
    return x_train, y_train, x_test, y_test

### Run experiments with standard DNN ###

## argv[1] - dataset to use (moons, digits, mnist)
## argv[2] - proportion of training data labeled

dataset = sys.argv[1]
labeled_proportion = float(sys.argv[2])

if dataset == 'moons':
    target = './data/moons.pkl'
    network = [100, 100]
    batchsize = 8
    n_epochs = 150
    learning_rate = 1e-3

elif dataset == 'digits':
    target = './data/digits.pkl'
    network = [150, 150]
    batchsize = 16
    n_epochs = 50
    learning_rate = 1e-3
 
with open(target, 'rb') as f:
    data = pickle.load(f)
x, y = data['x'], data['y']



Data = SSL_DATA(x,y, labeled_proportion=labeled_proportion) 
model = DNN(ARCHITECTURE=network, BATCH_SIZE=batchsize, NUM_EPOCHS=n_epochs, LEARNING_RATE=learning_rate)
model.fit(Data)


if dataset == 'moons':
    xl,yl = Data.data['x_l'], Data.data['y_l']
    x1 = xl[np.where(yl[:,1]==1)]
    x0 = xl[np.where(yl[:,0]==1)]

    X,Y = np.mgrid[-2:2.5:0.1, -2:2.5:0.1]
    xy = np.vstack((X.flatten(), Y.flatten())).T
    predictions = model.predict_new(xy.astype('float32'))


    range_vals = np.arange(-2.0,2.5,.1)
    zi = np.zeros(X.shape)
    for i, row_val in enumerate(range_vals):
        for j, col_val in enumerate(range_vals):
            idx = np.intersect1d(np.where(np.isclose(xy[:,0],row_val))[0],np.where(np.isclose(xy[:,1],col_val))[0])
            zi[i,j] = predictions[idx[0],0] * 100

    plt.contourf(X, Y, zi,cmap=plt.cm.coolwarm)

    plt.scatter(x1[:,0],x1[:,1], color='white')
    plt.scatter(x0[:,0],x0[:,1], color='black')

    plt.savefig('../experiments/Moons/contour_plot_dnn', bbox_inches='tight')
