from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import pdb, os

### Standard feedforward DNN for comparison ###

class DNN:
    """ Class defining our generative model """
    def __init__(self, LEARNING_RATE=0.05, ARCHITECTURE=[10], BATCH_SIZE=128, NUM_EPOCHS=10, nonlinearity=tf.nn.relu, logging=False):
    	## Step 1: define the placeholders for input and output
    	self.lr = LEARNING_RATE	          # learning rate
        self.ARCHITECTURE = ARCHITECTURE  # network architecture
	self.BATCH_SIZE = BATCH_SIZE      # batch size
	self.NUM_EPOCHS = NUM_EPOCHS      # number of training epochs
	self.NONLINEARITY = nonlinearity  # activation function
        self.LOGGING = logging            # log with TensorBoard

    def fit(self, Data):
    	self._process_data(Data)
    	
    	# Step 1: define the placeholders for input and output
    	self._create_placeholders()
    
       	## Step 2: define weights - setup all networks
        self._initialize_weights()
        
        ## Step 3: define the loss function
        y_ = self._forward_pass(self.x_batch)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_batch, logits=y_))
        
        ## Step 4: define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
	
	## Step 5: compute accuracies
	train_acc = self.compute_acc(self.x_train, self.y_train)
	test_acc  = self.compute_acc(self.x_test, self.y_test)
	batch_acc = self.compute_acc(self.x_batch, self.y_batch)
        
	## Step 6: initialize session and train
        SKIP_STEP, epoch = 50, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss, l_l, l_u, l_e = 0.0, 0.0, 0.0, 0.0
	    saver = tf.train.Saver()
	    if self.LOGGING:
                writer = tf.summary.FileWriter(self.LOGDIR, sess.graph)
 	    while epoch < self.NUM_EPOCHS:
                x_batch, y_batch, _, _ = Data.next_batch(self.BATCH_SIZE, self.BATCH_SIZE)
            	_, loss_batch = sess.run([self.optimizer, self.loss,], 
            			     	  feed_dict={self.x_batch: x_batch, 
		           		    	     self.y_batch: y_batch})
                total_loss += loss_batch
		#pdb.set_trace()
		if Data._epochs_labeled > epoch:
		    saver.save(sess, self.ckpt_dir, global_step=epoch+1)
		    epoch += 1
		    acc_train, acc_test, acc_batch  = sess.run([train_acc, test_acc, batch_acc],
						     		feed_dict = {self.x_train:Data.data['x_train'],
						     	 	  self.y_train:Data.data['y_train'],
								  self.x_test:Data.data['x_test'],
								  self.y_test:Data.data['y_test'],
								  self.x_batch:x_batch,
								  self.y_batch:y_batch})
		    print('At epoch {}: Training: {:5.3f}, Test: {:5.3f}, Labeled: {:5.3f}'.format(epoch, acc_train, acc_test, acc_batch))
	    if self.LOGGING:
	        writer.close()


    def predict(self, x):
	return tf.nn.softmax(self._forward_pass(x))

    def predict_new(self, x):
	predictions = self.predict(x)
	saver = tf.train.Saver()
	with tf.Session() as session:
	    ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
	    saver.restore(session, ckpt.model_checkpoint_path)
	    preds = session.run([predictions])
        return preds[0]

    def compute_acc(self, x, y):
	y_ = self.predict(x)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	 

    

    def _process_data(self, data):
    	""" Extract relevant information from data_gen """
    	self.TRAINING_SIZE = data.TRAIN_SIZE   			 # training set size
	self.TEST_SIZE = data.TEST_SIZE                          # test set size
	self.X_DIM = data.INPUT_DIM            			 # input dimension     
	self.NUM_CLASSES = data.NUM_CLASSES                      # number of classes
	self.data_name = data.NAME                               # set dataset name
	self._allocate_directory()                               # allocate directories
	


    def _create_placeholders(self):
 	""" Create input/output placeholders """
 	self.x_batch = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.X_DIM], name='x_batch')
 	self.y_batch = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.NUM_CLASSES], name='x_batch')
	self.x_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.X_DIM], name='x_train')
	self.y_train = tf.placeholder(tf.float32, shape=[self.TRAINING_SIZE, self.NUM_CLASSES], name='y_train')
	self.x_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.X_DIM], name='x_test')
	self.y_test = tf.placeholder(tf.float32, shape=[self.TEST_SIZE, self.NUM_CLASSES], name='y_test')
    	
    
    def _forward_pass(self, x):
	""" Forward pass through the network with weights as a dictionary """
	for i, neurons in enumerate(self.ARCHITECTURE):
	    weight_name, bias_name = 'W'+str(i), 'b'+str(i)
	    if i==0:
		h = self.NONLINEARITY(tf.add(tf.matmul(x, self.weights[weight_name]), self.weights[bias_name]))
	    else:
	        h = self.NONLINEARITY(tf.add(tf.matmul(h, self.weights[weight_name]), self.weights[bias_name]))
	out = tf.add(tf.matmul(h, self.weights['Wout']), self.weights['bout'])
	return out


    def _initialize_weights(self):
    	""" Initialize all model networks """
    	self.weights = {}
        for i, neurons in enumerate(self.ARCHITECTURE):
            weight_name, bias_name = 'W'+str(i), 'b'+str(i)
	    if i==0:
    		self.weights[weight_name] = tf.Variable(self._xavier_initializer(self.X_DIM, self.ARCHITECTURE[i]))
	    	self.weights[bias_name] = tf.Variable(tf.zeros([self.ARCHITECTURE[i]]))
	    else:
                self.weights[weight_name] = tf.Variable(self._xavier_initializer(self.ARCHITECTURE[i-1], self.ARCHITECTURE[i]))
                self.weights[bias_name] = tf.Variable(tf.zeros([self.ARCHITECTURE[i]]))
        self.weights['Wout'] = tf.Variable(self._xavier_initializer(self.ARCHITECTURE[-1], self.NUM_CLASSES))
        self.weights['bout'] = tf.Variable(tf.zeros([self.NUM_CLASSES]))


    
    def _xavier_initializer(self, fan_in, fan_out, constant=1): 
    	""" Xavier initialization of network weights"""
	low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    	high = constant*np.sqrt(6.0/(fan_in + fan_out))
    	return tf.random_uniform((fan_in, fan_out), 
        	                  minval=low, maxval=high, 
            	                  dtype=tf.float32)


    def _allocate_directory(self):
	self.LOGDIR =  'graphs/dnn-'+self.data_name+'-'+str(self.TRAINING_SIZE)+'/'
	self.ckpt_dir = './ckpt/dnn-'+self.data_name+'-'+str(self.TRAINING_SIZE)+'/'
	if not os.path.isdir(self.ckpt_dir):
	    os.mkdir(self.ckpt_dir)



