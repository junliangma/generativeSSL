from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, pickle, gzip, cPickle, pdb

import numpy as np
import tensorflow as tf

""" Class for mnist data to handle data loading and arranging """

class mnist:

    def __init__(self, path='data/mnist.pkl.gz', threshold=0.1):
	with gzip.open(path, 'rb') as f:
	    train_set, val_set, test_set = cPickle.load(f)
  	self.x_train, self.y_train = train_set[0], self.encode_onehot(train_set[1])
	if not len(val_set[0])==0:
  	    self.x_val, self.y_val = val_set[0], self.encode_onehot(val_set[1])
	    self.n_val = self.x_val.shape[0]
  	self.x_test, self.y_test = test_set[0], self.encode_onehot(test_set[1])
	self.n_train, self.n_test = self.x_train.shape[0], self.x_test.shape[0]
	self.drop_dimensions(threshold)
	self.x_dim, self.num_classes = self.x_train.shape[1], self.y_train.shape[1]

    def create_semisupervised(self, num_labels):
        x_u, y_u, x_l, y_l = [],[],[],[]
	for c in range(self.num_classes):
	    indices = np.where(self.y_train[:,c]==1)
	    xcls, ycls = self.x_train[indices], self.y_train[indices]
	    perm = np.random.permutation(xcls.shape[0])
	    x_l.append(xcls[:num_labels])
	    y_l.append(ycls[:num_labels])
	    x_u.append(xcls[num_labels:])
	    y_u.append(ycls[num_labels:])
	    self.x_labeled, self.y_labeled = np.concatenate(x_l), np.concatenate(y_l)
	    self.x_unlabeled, self.y_unlabeled = np.concatenate(x_u), np.concatenate(y_u)

    def drop_dimensions(self, threshold=0.1):
	stds = np.std(self.x_train, axis=0)
	good_dims = np.where(stds>threshold)[0]
	self.x_train = self.x_train[:,good_dims]
	if hasattr(self, 'x_val'):
	    self.x_val = self.x_val[:,good_dims]
	self.x_test = self.x_test[:,good_dims]

    def encode_onehot(self, labels):
	n, d = labels.shape[0], np.max(labels)+1
	return np.eye(d)[labels]
	
