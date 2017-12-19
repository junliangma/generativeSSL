import numpy as np
from scipy.io import loadmat
from data.SSL_DATA import SSL_DATA 
import pdb 

def loadDataSet(name, numLabeled):
    """ Load one of the benchmark datasets and split data
	name: name of the dataset (g241c, g241n, bci, coil, etc')
	numLabeled: number of labeled instances (10, 100)
    """
    nameStr = './data/benchmarkData/'+name
    data = loadmat(nameStr+'.mat')
    splits = loadmat(nameStr+'_l'+str(numLabeled)+'.mat')
    return (data, splits)

def encodeOneHot(labels):
    """ move from categorical to one-hot encoding """
    if len(labels.shape)>1:
	labels = np.squeeze(labels)
    labels[labels==-1]=0
    n, d = labels.shape[0], np.max(labels)+1
    return np.eye(d)[labels]

def sslFormat(dataDict):
    """ return an ssl data object from a data dictionary
	dictionary should contain 'x_u', 'y_u', 'x_l', 'y_l', 'name'
    """
    return SSL_DATA(dataDict['x_u'], dataDict['y_u'], 
		    x_labeled=dataDict['x_l'], y_labeled=dataDict['y_l'],
		    x_test=dataDict['x_u'], y_test=dataDict['y_u'],
		     dataset=dataDict['name']) 

def getBenchmarkSet(name, numLabeled, split):
    """ generate and return a data dictionary suitable for SSL experiments """
    data, splits = loadDataSet(name, numLabeled)
    x, y = data['X'], encodeOneHot(data['y'])
    idxL, idxU = splits['idxLabs'][split]-1, splits['idxUnls'][split]-1
    dataDict = {'x_l':x[idxL], 'y_l':y[idxL], 'x_u':x[idxU], 'y_u':y[idxU], 'name':name}
    return sslFormat(dataDict)

    
