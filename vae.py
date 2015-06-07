from __future__ import division, absolute_import
from __future__ import unicode_literals

import theano
import theano.tensor as T

import numpy as np
import lasagne

import vae
from utils.tensor_repeat import tensor_repeat

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

# Step 1: load data ####################################
import gzip, cPickle

f = gzip.open('mnist.pkl.gz', 'rb')
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = cPickle.load(f)
f.close()

print "X_train.shape: ", X_train.shape
print "X_train[0,:]", X_train[0,:]
 
# Step 0: initialization ##############################
num_data = X_train.shape[0]
batchsize = 100
L = 5
hidden_size = 400
z_size = 20
num_data_shared = theano.shared(np.array(num_data, dtype=theano.config.floatX), name='num_data')
batchsize_shared = theano.shared(np.array(batchsize, dtype=theano.config.floatX), name='batchsize')
L_shared = theano.shared(np.array(L, dtype=theano.config.floatX), name='L')

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
    #W = 0.0000001*np.arange(0,hidden_size*784).reshape((hidden_size,784)).T,
    #b = 0.0000001*np.arange(0,hidden_size).reshape((hidden_size,)),
) # batchsize x num_units

l_mu = lasagne.layers.DenseLayer(
    l_hidden1,
    num_units=z_size,
    nonlinearity=lasagne.nonlinearities.linear,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
    #W = 0.0000001*np.arange(0,z_size*hidden_size).reshape((z_size,hidden_size)).T,
    #b = 0.0000001*np.arange(0,z_size).reshape((z_size,)),
) # batchsize x num_units

l_log_sigma = lasagne.layers.DenseLayer(
    l_hidden1, 
    num_units=z_size,
    nonlinearity=lasagne.nonlinearities.linear,
    W=lasagne.init.Normal(std=0.01 * 0.5), b=lasagne.init.Normal(std=0.01 * 0.5),
    #W = 0.0000001*np.arange(0,z_size*hidden_size).reshape((z_size,hidden_size)).T,
    #b = 0.0000001*np.arange(0,z_size).reshape((z_size,)),
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
# 
# - fully connected layer - hidden_size units (from z to h)
# - tanh (for h)
l_hidden2 = lasagne.layers.DenseLayer(
    l_z,
    num_units=hidden_size, 
    nonlinearity=lasagne.nonlinearities.tanh,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
    #W = 0.0000001*np.arange(0,hidden_size*z_size).reshape((hidden_size,z_size)).T,
    #b = 0.0000001*np.arange(0,hidden_size).reshape((hidden_size,)),
)

# - fully connected layer - 784 units (from h2 to x_out)
# - sigmoid
l_out = lasagne.layers.DenseLayer(
    l_hidden2, 
    num_units=28*28, 
    nonlinearity=lasagne.nonlinearities.sigmoid,
    W=lasagne.init.Normal(std=0.01), b=lasagne.init.Normal(std=0.01),
    #W = 0.0000001*np.arange(0,hidden_size*784).reshape((784,hidden_size)).T,
    #b = 0.0000001*np.arange(0,784).reshape((784,)),
)

# build cost
#batchsize = lasagne.layers.get_output(l_in).shape[0]
input_dim = T.prod(lasagne.layers.get_output(l_in).shape[1:])
input_tmp =  lasagne.layers.get_output(l_in).reshape((lasagne.layers.get_output(l_in).shape[0], T.prod(lasagne.layers.get_output(l_in).shape[1:])))
'''
input_tmp_list = []
for i in xrange(L):
    input_tmp_list.append(theano.tensor.ones_like(input_tmp))
    input_tmp_list[i] = input_tmp_list[i] * input_tmp
input = theano.tensor.concatenate(input_tmp_list, axis=1)
'''
input = tensor_repeat(input_tmp, size=L, axis=1)
input = input.reshape((batchsize * L, input_dim))
output =  lasagne.layers.get_output(l_out)
logpxz = -T.nnet.binary_crossentropy(output, input).sum() / L 

minus_kl_div = 0.5 * (1 + 2*lasagne.layers.get_output(l_log_sigma)- theano.tensor.sqr(lasagne.layers.get_output(l_mu)) - theano.tensor.exp(2*lasagne.layers.get_output(l_log_sigma))).sum()

lower_bound = (minus_kl_div + logpxz) #/batchsize

# calculate gradient
all_params = lasagne.layers.get_all_params(l_out)
all_grads = theano.grad(lower_bound, all_params) 

## temp
grad_fn = theano.function(inputs=[l_in.input_var],
                          outputs=all_grads)
minus_kl_div_fn = theano.function(inputs=[l_in.input_var],
                          outputs=[minus_kl_div])
logpxz_fn = theano.function(inputs=[l_in.input_var],
                          outputs=[logpxz])
all_layers_fn = theano.function(inputs=[l_in.input_var], 
                          outputs=[lasagne.layers.get_output(l_in), lasagne.layers.get_output(l_hidden1), lasagne.layers.get_output(l_mu), lasagne.layers.get_output(l_log_sigma), lasagne.layers.get_output(l_z), lasagne.layers.get_output(l_hidden2), lasagne.layers.get_output(l_out)])  

'''
updates = lasagne.updates.momentum(
    loss_or_grads=all_grads,
    params=all_params,
    learning_rate=0.001,
    momentum=1/500)
updates = lasagne.updates.adagrad(
    loss_or_grads=all_grads,
    params=all_params,
    learning_rate=0.01,
)
'''


params_asdf = lasagne.layers.get_all_param_values(l_out)
h_asdf = [0.01] * len(params_asdf)


print len(all_params)
print all_params
print all_params[0].type 
prior = np.array([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]) * all_params
#raise NameError("hi")
learning_rate = 0.01
epsilon = 0. #1e-6

from collections import OrderedDict

updates = OrderedDict()

for param, grad, p in zip(all_params, all_grads, prior):
    value = param.get_value(borrow=True)
    accu = theano.shared(0.01 * np.ones(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
    accu_new = accu + grad ** 2
    updates[accu] = accu_new
    updates[param] = param + (learning_rate * (grad - p * param * np.array(batchsize, dtype=value.dtype) / np.array(num_data, dtype=value.dtype)) /
                              theano.tensor.sqrt(accu_new))


#all_params_new = all_params + learning_rate / theano.tensor.sqrt(theano.tensor.sqr(all_grads) + epsilon) * (all_grads - prior * batchsize / num_data)
#updates = (all_params, all_params_new)
#all_params += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))


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

# ################################# training #################################

print("Starting training...")

num_epochs = 100
for epoch_num in range(num_epochs):
    # iterate over training minibatches and update the weights
    num_batches_train = int(np.ceil(len(X_train) / batchsize))
    train_losses = []
    for batch_num in range(num_batches_train):
        batch_slice = slice(batchsize * batch_num,
                            batchsize * (batch_num + 1))
        X_batch = X_train[batch_slice]
        y_batch = y_train[batch_slice]
        
        grads = grad_fn(X_batch)
        '''
        for (param, grad) in zip(all_params, grads):
            print param
            print grad
        val11, = minus_kl_div_fn(X_batch)
        val22, = logpxz_fn(X_batch)
        [val1,val2,val3,val4,val5,val6,val7] = all_layers_fn(X_batch)
        #print "batch: ", X_batch[0,0,:,:]
        #print "grad :", grad
        #print "minus_kl_div :", val11
        #print "logpxz :", val22
        print "\n\n\n\n\n\n"
        print "l_in: ", val1
        print "l_hidden1: ", val2
        print "l_mu: ", val3
        print "l_log_sigma: ", val4
        print "l_z: ", val5
        print "l_hidden2: ", val6
        print "l_out: ", val7
        ''' 
        
        #loss, = train_fn(X_batch, y_batch)
        loss, = train_fn(X_batch)
         
        #print "minus_kl_div :", val11
        #print "logpxz :", val22
        #print "loss :", loss
        ''' 
        params_updated = lasagne.layers.get_all_param_values(l_out)
        for i in xrange(len(params_updated)):
            print "asdf ", all_params[i]
            print params_updated[i]
            print batchsize
            print num_data
         
        for i in xrange(len(params_asdf)):
            h_asdf[i] += np.asarray(grads[i])*np.asarray(grads[i])
            #if i < 5 or (i < 6 and len(params_asdf) == 12):
            if i % 2 is 0:
                prior_asdf = 0.5*params_asdf[i]
            else:
                prior_asdf = 0
            #print "self.params[i].shape: ", self.params[i].shape
            #print "totalGrandients[i].shape: ", totalGradients[i].shape
            params_asdf[i] += learning_rate/np.sqrt(h_asdf[i]) * (np.asarray(grads[i]) - prior_asdf*(batchsize/num_data))
            #print all_params[i] 
            #print params_asdf[i]
            #print batchsize
            #print num_data

        lasagne.layers.set_all_param_values(l_out, params_asdf)
         
        
        raise NameError("Hi")       
        if np.isnan(loss).sum() >= 1:
            raise ValueError('Nan in loss')
        ''' 
        train_losses.append(loss/batchsize)
    # aggregate training losses for each minibatch into scalar
    train_loss = np.mean(train_losses)

    # calculate validation loss
    num_batches_valid = int(np.ceil(len(X_valid) / batchsize))
    valid_losses = []
    list_of_probabilities_batch = []
    for batch_num in range(num_batches_valid):
        batch_slice = slice(batchsize * batch_num,
                            batchsize * (batch_num + 1))
        X_batch = X_valid[batch_slice]
        y_batch = y_valid[batch_slice]

        #loss, probabilities_batch = valid_fn(X_batch, y_batch)
        loss, probabilities_batch = valid_fn(X_batch)
        #print(probabilities_batch.shape)
        #raise NameError('Hi There')
        valid_losses.append(loss/batchsize)
        list_of_probabilities_batch.append(probabilities_batch)
    valid_loss = np.mean(valid_losses)
    # concatenate probabilities for each batch into a matrix
    probabilities = np.concatenate(list_of_probabilities_batch)
    # calculate classes from the probabilities
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    #accuracy = sklearn.metrics.accuracy_score(y_valid, predicted_classes)

    print("Epoch: %d, train_loss=%f, valid_loss=%f"
          % (epoch_num + 1, train_loss, valid_loss))
    #print("Epoch: %d, train_loss=%f, valid_loss=%f, valid_accuracy=%f"
    #      % (epoch_num + 1, train_loss, valid_loss, accuracy))
