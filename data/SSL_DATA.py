from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pdb

""" Generate a data structure to support SSL models. Expects:
x - np array: N rows, d columns
y - np array: N rows, k columns (one-hot encoding)
"""


class SSL_DATA:
    """ Class for appropriate data structures """
    def __init__(x, y, train_proportion=0.7, labeled_proportion=0.1, 
		     LABELED_BATCHSIZE=7, UNLABELED_BATCHSIZE=63, dataset='moons'):
	self.N = x.shape[0] 
	self.INPUT_DIM = x.shape[1]
	self.NUM_CLASSES = y.shape[1]
	self.NAME = dataset
	self.TRAIN_SIZE = np.round(train_proportion * self.NUM_TRAIN)
	self.NUM_LABELED = np.round(labeled_proportion * self.TRAIN_SIZE)
	self.NUM_UNLABELED = self.TRAIN_SIZE - self.NUM_LABELED
	self.LABELED_BATCHSIZE = LABELED_BATCHSIZE
	self.UNLABELED_BATCHSIZE = UNLABELED_BATCHSIZE

	# create necessary data splits
	xtrain, ytrain, xtest, ytest = _split_data(x,y)
	x_labeled, y_labeled, x_unlabeled, y_unlabeled = _create_semisupervised(xtrain, ytrain)

	# create appropriate data dictionaries
	self.data = {}
	self.data['x_u'], self.data['y_u'] = x_unlabeled, y_unlabeled
	self.data['x_l'], self.data['y_l'] = x_labeled, y_labeled
	self.data['x_test'], self.data['y_test'] = xtest, ytest

	# counters and indices for minibatching
	self._start_labeled, self._start_unlabeled = 0, 0
	self._epochs_labeled = 0
	self._epochs_unlabeled = 0

    def _split_data(self, x, y):
	""" split the data according to the proportions """
	indices = np.random.shuffle(range(self.N))
	train_idx, test_idx = indices[:self.TRAIN_SIZE], indices[self.TRAIN_SIZE:]
	return (x[train_idx,:], y[train_idx,:], x[test_idx,:], y[test_idx,:])

    def _create_semisupervised(self, x, y):
	""" split training data into labeled and unlabeled """
	indices = np.random.shuffle(range(self.TRAIN_SIZE))
	l_idx, u_idx = indices[:self.NUM_LABELED], indices[self.NUM_LABELED:]
	return (x[l_idx,:], y[l_idx,:], x[u_idx,:], y[u_idx,:])


    def next_batch(self):
    	x_l_batch, y_l_batch = self.next_batch_labeled()
    	x_u_batch, y_u_batch = self.next_batch_unlabeled()
    	return (x_l_batch, y_l_batch, x_u_batch, y_u_batch)


    def next_batch_labeled(self, shuffle=True):
    	"""Return the next `batch_size` examples from this data set."""
        start = self._start_labeled
    	# Shuffle for the first epoch
    	if self._epochs_labeled == 0 and start == 0 and shuffle:
      	    perm0 = np.random.shuffle(np.arange(self.NUM_LABELED))      		
      	    self.data['x_l'], self.data['y_l'] = self.data['x_l'][perm0,:], self.data['y_l'][perm0,:]
   	# Go to the next epoch
    	if start + self.LABELED_BATCHSIZE > self.NUM_LABELED:
      	    # Finished epoch
      	    self._epochs_labeled += 1
      	    # Get the rest examples in this epoch
      	    rest_num_examples = self.NUM_LABELED - start
      	    inputs_rest_part = self.data['x_l'][start:self.NUM_LABELED]
      	    labels_rest_part = self.data['y_l'][start:self.NUM_LABELED]
      	    # Shuffle the data
      	    if shuffle:
                perm = np.random.shuffle(np.arange(self.NUM_LABELED))
        	self.data['x_l'] = self.data['x_l'][perm]
        	self.data['y_l'] = self.data['y_l'][perm]
      	    # Start next epoch
      	    start = 0
      	    self._start_labeled = self.LABELED_BATCHSIZE - rest_num_examples
      	    end = self._start_labeled
      	    inputs_new_part = self.data['x_l'][start:end]
      	    labels_new_part = self.data['y_l'][start:end]
      	    return np.concatenate((inputs_rest_part, inputs_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    	else:
      	    self._start_labeled += self.LABELED_BATCHSIZE
      	    end = self._start_labeled
	    return self.data['x_l'][start:end], self.data['y_l'][start:end]

	

    def next_batch_unlabeled(self, shuffle=True):
    	"""Return the next `batch_size` examples from this data set."""
        start = self._start_unlabeled
    	# Shuffle for the first epoch
    	if self._epochs_unlabeled == 0 and start == 0 and shuffle:
      	    perm0 = np.random.shuffle(np.arange(self.NUM_UNLABELED))      		
      	    self.data['x_u'], self.data['y_u'] = self.data['x_u'][perm0,:], self.data['y_u'][perm0,:]
   	# Go to the next epoch
    	if start + self.UNLABELED_BATCHSIZE > self.NUM_UNLABELED:
      	    # Finished epoch
      	    self._epochs_unlabeled += 1
      	    # Get the rest examples in this epoch
      	    rest_num_examples = self.NUM_UNLABELED - start
      	    inputs_rest_part = self.data['x_u'][start:self.NUM_UNLABELED]
      	    labels_rest_part = self.data['y_u'][start:self.NUM_UNLABELED]
      	    # Shuffle the data
      	    if shuffle:
        	perm = np.random.shuffle(np.arange(self.NUM_UNLABELED))
        	self.data['x_u'] = self.data['x_u'][perm]
        	self.data['y_u'] = self.data['y_u'][perm]
      	    # Start next epoch
      	    start = 0
      	    self._start_unlabeled = self.UNLABELED_BATCHSIZE - rest_num_examples
      	    end = self._start_unlabeled
      	    inputs_new_part = self.data['x_u'][start:end]
      	    labels_new_part = self.data['y_u'][start:end]
      	    return np.concatenate((inputs_rest_part, inputs_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    	else:
      	    self._start_unlabeled += self.LABELED_UNBATCHSIZE
      	    end = self._start_unlabeled
	    return self.data['x_u'][start:end], self.data['y_u'][start:end]




