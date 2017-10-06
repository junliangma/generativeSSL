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
from models.aux_gssl2 import agssl
from models.sdgssl import sdgssl
from models.adgm import adgm

### Script to run an experiment with generative SSL model ###

## argv[1] - dataset to use (moons, digits, mnist)
## argv[2] - proportion of training data labeled (or for mnist, number of labels from each class)
## argv[3] - Dataset seed
## argv[4] - noise level in moons dataset / Threshold for reduction in mnist
## argv[5] - model to use (adgm, adgssl, sdgssl)


dataset, noise = sys.argv[1], sys.argv[4]
seed = int(sys.argv[3])
modelName = sys.argv[5]


if dataset == 'moons':
    target = './data/moons_'+noise+'.pkl'
    labeled_proportion = float(sys.argv[2])
    labeled_batchsize, unlabeled_batchsize = 4,128
    
    z_dim, a_dim = 5,3
    learning_rate = (3e-4,300)
    architecture = [128,128]
    n_epochs = 100
    temperature_epochs = 99
    start_temp = 0.0  

    verbose=1
    binarize = False
    logging = False


    num_labeled, threshold = int(sys.argv[2]), float(noise)
    labeled_batchsize, unlabeled_batchsize = 100,100
    data = mnist(target, threshold=threshold)
    data.create_semisupervised(num_labeled)    

    z_dim, a_dim = 100, 100
    learning_rate = (3e-4,)
    architecture = [500, 500]
    n_epochs = 500
    type_px = 'Bernoulli'
    temperature_epochs, start_temp = None, 0.0
    l2_reg = 0.0
    batchnorm = False
    binarize, logging = True, False
    verbose = 1

if dataset in ['moons', 'digits']:
    with open(target, 'rb') as f:
        data = pickle.load(f)
    x, y = data['x'], data['y']
    data = SSL_DATA(x,y, labeled_proportion=labeled_proportion, dataset=dataset, seed=seed) 
elif dataset == 'mnist':
    data = SSL_DATA(data.x_unlabeled, data.y_unlabeled, x_test=data.x_test, y_test=data.y_test, 
		    x_labeled=data.x_labeled, y_labeled=data.y_labeled, dataset=dataset, seed=seed)


if modelName=='agssl':
	model = agssl(Z_DIM=z_dim, A_DIM=a_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, 
			BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp, eval_samps=2000,
			l2_reg=l2_reg, BATCHNORM=batchnorm, LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, 
			verbose=verbose, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)

elif modelName=='sdgssl':
	model = sdssl(Z_DIM=z_dim, A_DIM=a_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, 
			BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp, eval_samps=2000,
			l2_reg=l2_reg, BATCHNORM=batchnorm, LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, 
			verbose=verbose, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)

elif modelName=='adgm':
	model = adgm(Z_DIM=z_dim, A_DIM=a_dim, LEARNING_RATE=learning_rate, NUM_HIDDEN=architecture, ALPHA=0.1, 
			BINARIZE=binarize, temperature_epochs=temperature_epochs, start_temp=start_temp, eval_samps=2000,
			l2_reg=l2_reg, BATCHNORM=batchnorm, LABELED_BATCH_SIZE=labeled_batchsize, UNLABELED_BATCH_SIZE=unlabeled_batchsize, 
			verbose=verbose, NUM_EPOCHS=n_epochs, TYPE_PX=type_px, logging=logging)

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
    plt.scatter(xl[:,0],xl[:,1], color='black', s=4)
    plt.savefig('../experiments/Moons/trial', bbox_inches='tight')

    plt.figure()
    x,y = model._sample_xy(int(1e4))
    y_bin = np.argmax(y, axis=1)
    x0, x1 = x[np.where(y_bin==0)], x[np.where(y_bin==1)]

    plt.scatter(x0[:,0],x0[:,1], s=1, color='r')
    plt.scatter(x1[:,0],x1[:,1], s=1, color='b')
    plt.savefig('../experiments/Moons/sample_trial', bbox_inches='tight')

    

if dataset=='mnist':
    preds_test = model.predict_new(data.data['x_test'].astype('float32'))
    acc, ll = np.mean(np.argmax(preds_test,1)==np.argmax(data.data['y_test'],1)), -log_loss(data.data['y_test'], preds_test)
    print('Test Accuracy: {:5.3f}, Test log-likelihood: {:5.3f}'.format(acc, ll))

    ## t-sne visualization
    cl = plt.cm.tab10(np.linspace(0,1,10))
    test_labs = np.argmax(data.data['y_test'], 1)
    z_test = model.encode_new(data.data['x_test'].astype('float32'))
    np.save('./z_ssple', z_test)
    np.save('./test_labs_sslpe', test_labs)
    #t = tsne(n_components=2, random_state=0)
    #print('Starting TSNE transform for latent encoding...')
    #sslpe_reduced = t.fit_transform(z_test)
    #print('Done with TSNE transformations...')
    #plt.figure(figsize=(8,10), frameon=False)
    #for digit in range(10):
    #    indices = np.where(test_labs==digit)[0]
    #    plt.scatter(sslpe_reduced[indices, 0], sslpe_reduced[indices, 1], c=cl[digit], label='Digit: '+str(digit))
    #plt.legend()
    #plt.savefig('mnist_samps/sslpe_encode', bbox_inches='tight')

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


