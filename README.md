# generativeSSL
Deep generative model for labels for semi-supervised learning.
Model development for Cambridge masters' thesis.

Next on the TO-DO list:
1. Verify performance of VAE
2. Consider MNIST

Note that currently:
1. VAE only supports Gaussian distributions on the inputs
2. Models only interact with data from class SSL_DATA.
3. Running experiments assumes that the data (moons, digits) is pickled in a dictionary in the data folder.
