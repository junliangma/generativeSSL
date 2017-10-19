Deep Generatives Models for Semi-Supervised Learning
=======
This repository implements a number of DGMs for semi-supervised learning in tensorflow. The repository contains implementations of:

* M2 - `Semisupervised learning with deep generative models <https://arxiv.org/abs/1406.5298?>`_.
* ADGM, SDGM - `Auxiliary deep generative models <http://arxiv.org/abs/1602.05473>`_.
* SSLPE, SSLAPD - `Bayesian semisupervised learning with deep generative models >https://arxiv.org/abs/1706.09751>`_.


The models are implementated in `TensorFlow  1.3 <https://www.tensorflow.org/api_docs/>`_.


Installation
------------
Please make sure you have installed the requirements before executing the python scripts.


**Install**


.. code-block:: bash

  pip install scipy
  pip install numpy
  pip install matplotlib
  pip install tensorflow(-gpu)

Examples
-------------
The repository primarily includes a script running a new model on the MNIST dataset with only 100 labels - *run_sdgmssl_mnist.py*.

Please see the source code and code examples for further details. For some visualisations of the latent space and the
half moon classification examples, see https://youtu.be/g-c-xOmA2nA, https://youtu.be/hnhkKTSdmls and https://youtu.be/O8-VYr4CxsI.

The repository includes a template for designing further deep generative models. The templates make use of the library found in *utils/dgm.py* which contains Bayesian deep learning functionalities. An example script (*run_mnist.py*) will also be included which demonstrates how to successfully use the models and interact with the data wrapper found in *data/SSL_DATA.py*. The script trains a model of choice <*M2, ADGM, SDGM, (aux, skip)-SSLPE, (aux, skip)-SSLAPD*> using only 10 labeled examples form each class in MNIST, and should be run:
.. code-block:: bash
  python run_mnist.py <m2, adgm, sdgm, sslpe, sslapd,...>
