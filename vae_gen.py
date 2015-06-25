from __future__ import division, absolute_import
from __future__ import unicode_literals

import theano
import theano.tensor as T

import numpy as np
import lasagne

import vae
from utils.tensor_repeat import tensor_repeat
from utils.updates import adagrad_w_prior

# pickle
try:
    import cPickle as pickle
except:
    #import pickle
    print("hi")

# Variational Inference 
# let us have p(x|z) and p(z) (this can be true prob dist or some models we assumed)
# and we want to find p(z|x) 
#
# KL(q||p) = sigma_{z}{q(z|x) log{ q(z|x) / p(z|x) }}
#          = sigma_{z}{q(z|x) log{ q(z|x)/(p(x,z)/p(x)) }}
#          = sigma_{z}{q(z|x) log{ q(z|x)p(x) / p(x,z) }}
#          = sigma_{z}{q(z|x) (log{ q(z|x)/p(x,z) } + log{p(x)}}) 
#          = sigma_{z}{q(z|x) log{ q(z|x)/p(x,z) }} + log{p(x)}
# -> log{p(x)} = KL(q||p) - sigma_{z}{q(z|x)log{q(z|x)/p(x,z)}}
#              = KL(q||p) + L(q) where L(q) = - sigma_{z}{q(z|x)log{q(z|x)/p(x,z)}}
# -> by optimizing(maximizing) L(q), we could have q(z|x) that are close to p(z|x)
# 
# Stochastic Gradient Variational Bayes
# L(q) = -sigma_{z}{q(z|x)log{q(z|x)/p(x,z)}}
#      = sigma_{z}{q(z|x)( -log{q(z|x)} + log{p(x,z)} )}
#      = sigma_{z}{q(z|x)( -log{q(z|x)} + log{p(x|z)p(z)} )}
#      = sigma_{z}{q(z|x)( -log{q(z|x)/p(z)} + log{p(x|z)}}
#      = -sigma_{z}{q(z|x)log{q(z|x)/p(z)}} + sigma_{z}{q(z|x)log{p(x|z)}}
#       
# in here p(z|x) = true posterior
#         q(z|x) = encoder (i.e. approximate posterior) 
#         p(x|z) = decoder (i.e. likelihood)
#
# variational autoencoder
#     q(z|x) to be neural networks
#     p(x|z) to be neural networks 
# 
# in this example, we use specifically
# p(z) = univariate Gaussian
# q(z|x) = g(e,x) where e ~ N(0,1)
# thus we can use reparametrization trick with e.q. 8


# Step 0: load data ####################################
import gzip, cPickle

f = gzip.open('mnist.pkl.gz', 'rb')
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = cPickle.load(f)
f.close()
#print "X_train.shape: ", X_train.shape
#print "X_train[0,:]", X_train[0,:]

 
# Step 1: initialization ##############################
num_data = X_train.shape[0]
batchsize = 100
L = 1
hidden_size = 400
z_size = 200
update_rules = 'adagrad' # you can choose either 1) momentum, 2) adagrad, and 3) adagrad_w_prior. 
num_epochs = 2000


# Step 2: build model -> equals to build model #########
# architecture as follows; 
# encoder: 
# - fully connected layer - hidden_size units (from x to h)
# - tanh (for h)
# - fully connected layer - z_size units (from h to mu)
# - fully connected layer - z_size units (from h to sigma_sq)
#
# - input layer
l_in = lasagne.layers.InputLayer(
    shape=(batchsize, 28*28)
)

# - fully connected layer - hidden_size units (from x to h)
# - tanh (for h)
l_hidden1 = lasagne.layers.DenseLayer(
    l_in, 
    num_units=hidden_size,
    nonlinearity=lasagne.nonlinearities.tanh,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
) # batchsize x num_units

l_mu = lasagne.layers.DenseLayer(
    l_hidden1,
    num_units=z_size,
    nonlinearity=lasagne.nonlinearities.linear,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
) # batchsize x num_units

l_log_sigma = lasagne.layers.DenseLayer(
    l_hidden1, 
    num_units=z_size,
    nonlinearity=lasagne.nonlinearities.linear,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
) # batchsize x num_units

l_z = vae.layers.GaussianPropLayer(
    l_mu, 
    l_log_sigma, 
    L,
)

# decoder:
# - fully connected layer - hidden_size units (from z to h)
# - tanh (for h)
# - fully connected layer - 784 units (from h to x)
# - sigmoid
#
# - fully connected layer - hidden_size units (from z to h)
# - tanh (for h)
l_hidden2 = lasagne.layers.DenseLayer(
    l_z,
    num_units=hidden_size, 
    nonlinearity=lasagne.nonlinearities.tanh,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
)

# - fully connected layer - 784 units (from h2 to x_out)
# - sigmoid
l_out = lasagne.layers.DenseLayer(
    l_hidden2, 
    num_units=28*28, 
    nonlinearity=lasagne.nonlinearities.sigmoid,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
)


# decoder decoder
l_in_tmp = lasagne.layers.InputLayer(
    shape=(batchsize, z_size)
)
l_z_tmp = lasagne.layers.GaussianNoiseLayer(
    l_in_tmp,
    sigma=1,
)

l_hidden2_tmp = vae.layers.DenseLayerDummy(
    l_z_tmp,
    origin=l_hidden2,
)

l_out_tmp = vae.layers.DenseLayerDummy(
    l_hidden2_tmp,
    origin=l_out,
)

# - build cost
input_dim = T.prod(lasagne.layers.get_output(l_in).shape[1:])
input_tmp =  lasagne.layers.get_output(l_in).reshape((lasagne.layers.get_output(l_in).shape[0], T.prod(lasagne.layers.get_output(l_in).shape[1:])))
input = tensor_repeat(input_tmp, size=L, axis=1)
input = input.reshape((batchsize * L, input_dim))
output =  lasagne.layers.get_output(l_out)
logpxz = -T.nnet.binary_crossentropy(output, input).sum() / L 

minus_kl_div = 0.5 * (1 + 2*lasagne.layers.get_output(l_log_sigma)- theano.tensor.sqr(lasagne.layers.get_output(l_mu)) - theano.tensor.exp(2*lasagne.layers.get_output(l_log_sigma))).sum()

lower_bound = (minus_kl_div + logpxz) #/batchsize

# - calculate gradient
# - all update rule implementations fitted to gradient descent (cost minimization), but what we want to do is to maximize lower_bound. 
#   Thus, multiply minus one to the cost. 
all_params = lasagne.layers.get_all_params(l_out)
all_grads = theano.grad(-lower_bound, all_params)
'''
# - update rules 
#   you can choose update rules 
#   1) momentum, 2) adagrad, and 3) adagrad_w_prior
# - momentum is not really work for vae (exploding easily)
if update_rules == 'momentum':
    updates = lasagne.updates.momentum(
        loss_or_grads=all_grads,
        params=all_params,
        learning_rate=0.000001,
        momentum=0.)
elif update_rules == 'adagrad':
    updates = lasagne.updates.adagrad(
        loss_or_grads=all_grads,
        params=all_params,
        learning_rate=0.01,
    )
elif update_rules == 'adagrad_w_prior':
    updates = adagrad_w_prior(
        loss_or_grads=all_grads,
        params=all_params,
        learning_rate=0.01,
        batchsize=batchsize,
        num_data=num_data,
    )
else:
    raise ValueError('Please specify learning rule')

# - create a function that also updates the weights
# - this function takes in 2 arguments: the input batch of images and a
#   target vector (the y's) and returns a list with a single scalar
#   element (the loss)
train_fn = theano.function(inputs=[l_in.input_var],
                           outputs=[lower_bound],
                           updates=updates)

# - create a function that does not update the weights, and doesn't
#   use dropout
# - same interface as previous the previous function, but now the
#   output is a list where the first element is the loss, and the
#   second element is the actual predicted probabilities for the
#   input data
valid_fn = theano.function(inputs=[l_in.input_var],
                           outputs=[lower_bound,
                                    lasagne.layers.get_output(l_out, deterministic=True)])

'''
encode_fn = theano.function(inputs=[l_in.input_var],
                           outputs=[lasagne.layers.get_output(l_z, deterministic=True)])
recon_fn = theano.function(inputs=[l_in_tmp.input_var],
                           outputs=[lasagne.layers.get_output(l_out_tmp, deterministic=True)])

# ################################# training #################################

# load
epoch_num = 1900
weights_load = pickle.load( open( "mnist_vae_h_%d_z_%d_epoch_%d.weight.pkl" % (hidden_size, z_size, epoch_num), "rb" ) )
lasagne.layers.set_all_param_values(l_out, weights_load)

print y_train

recon, = recon_fn(np.random.rand(batchsize, z_size).astype(theano.config.floatX))

print recon.shape
import cv2
for i in xrange(recon.shape[0]):
    image = recon[i,:]
    image = 255 * image.reshape((28, 28))
    cv2.imwrite('recon_%d.png' % i, image)

