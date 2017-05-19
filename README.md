# generativeSSL
Deep generative model for labels for semi-supervised learning.
Model development for Cambridge masters' thesis.

Next on the TO-DO list:
1. Implement Kingma's model
2. Release constraint on network architecture (scale up networks)
3. Make sure everything works for digits dataset as well

Note that currently:
1. VAE only supports Gaussian distributions on the inputs
2. GenerativeSSL only supports networks that are 1-layer (all have same architecture)
3. Models only interact with data from class SSL_DATA.
