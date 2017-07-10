# generativeSSL
Deep generative model for labels for semi-supervised learning.
Model development for Cambridge masters' thesis.

Next on the TO-DO list:
1. Add batch normalization functionality to Bayesian version 
2. Run active learning experiments for Moons data

Note that currently:
1. Batch normalization is only implemented for GSSL model.
2. Models only interact with data from class SSL_DATA.
3. Running experiments assumes that the data (moons, digits) is pickled in a dictionary in the data folder.
