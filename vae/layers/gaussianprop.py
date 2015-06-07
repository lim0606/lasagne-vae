import numpy as np
import theano
#import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.layers.base import MergeLayer #from .base import MergeLayer

__all__ = [
    "GaussianPropLayer",
]


class GaussianPropLayer(MergeLayer):
    """
    lasagne.layers.GaussianPropLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming_mu, incoming_sigma_sq, L=10, **kwargs):
        super(GaussianPropLayer, self).__init__(incomings=[incoming_mu, incoming_sigma_sq], **kwargs)
        self.L = L
        if self.input_shapes[0][0] is not self.input_shapes[1][0]:
            raise ValueError("Mismatch: input shape of mu and sigma_sq are mismatched")

        self.srng = RandomStreams(seed=234)
                
    def get_output_shape_for(self, input_shapes):
        #print input_shapes[0][0]
        return (input_shapes[0][0] * self.L, input_shapes[0][1])

    def get_output_for(self, inputs, **kwargs):
        for input in inputs:
            if input.ndim > 2:
                # if the input has more than two dimensions, flatten it into a
                # batch of feature vectors.
                input = input.flatten(2)

        #print "self.input_shapes[0][0]", self.input_shapes[0][0]
        #print "self.input_shapes[0][1]", self.input_shapes[0][1]

        #self.eta = self.srng.normal((self.input_shapes[0][0] * self.L, self.input_shapes[0][1]))
        self.eta = self.srng.normal((self.input_shapes[0][0], self.input_shapes[0][1]))
        # input_shapes[0] = mu.shape
        # input_shapes[1] = log_sigma.shape
        # input_shapes[0][0] = batchsize
        # input_shapes[0][1] = num_inputs
        #eta_printed = theano.printing.Print('eta')(self.eta)        

        # inputs[0] = mu
        # inputs[1] = log_sigma i.e. log(sigma)
        #print "inputs[0].shape[0]", inputs[0].shape[0]
        #print "inputs[0].shape[1]", inputs[0].shape[1]
        #print "inputs[0].ndim", inputs[0].ndim
        #inputs_0_printed = theano.printing.Print('inputs[0]')(inputs[0])

        z_tmp = inputs[0] + theano.tensor.exp(inputs[1]) * self.eta # * eta_printed

        z_tmp_list = []
        for i in xrange(self.L):
            z_tmp_list.append(theano.tensor.ones_like(z_tmp))
            z_tmp_list[i] = z_tmp_list[i] * z_tmp 
        z = theano.tensor.concatenate(z_tmp_list, axis=1)
        # from utils.tensor_repeat import tensor_repeat
        # z = tensor_repeat(z_tmp, size=self.L, axis=1)

        z = z.reshape((self.input_shapes[0][0] * self.L, self.input_shapes[0][1]))

        return z

