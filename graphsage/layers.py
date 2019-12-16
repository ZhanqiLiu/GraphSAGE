from __future__ import division
from __future__ import print_function

import tensorflow as tf

from graphsage.utils import zeros

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
        return self.act(output)

class Bilstm(Layer):
    """Dense layer."""
    def __init__(self,feature_length,dropout=0.,**kwargs):
        super(Bilstm, self).__init__(**kwargs)

        self.dropout = dropout
        self.feature_length = feature_length


        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        X_initial = tf.stack(inputs, axis=0)
        X = tf.reshape(X_initial , [-1, len(inputs), self.feature_length])

        with tf.variable_scope(self.name + '_vars'):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=16, state_is_tuple=True)#tf.shape(Layer[1])[1]
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout)

            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=16, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X, dtype=tf.float32)#[(?,3,16),(?,3,16)]

            outputs_concat = tf.concat([outputs[0], outputs[1]], axis=2)#(?,3,32)
            concatenated = tf.layers.dense(outputs_concat, units=1, use_bias=True, kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False))#(?,3,1)
            #outputs_concat = tf.concat([outputs[0][-1], outputs[1][-1]], axis=1)  # (?,3,32)
            #concatenated = tf.layers.dense(outputs_concat, units=3, use_bias=True, kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False))#(?,3,1)


        #一个w，有多少条数据都可以进来，最后对每个样本点都生成一个3维向量，不能做平均
        bacth_average = tf.math.reduce_mean(tf.reshape(concatenated, [-1, len(inputs)]),axis=0)

        # 可以选择：concatenated，bacth_average
        weight_each_vertex = bacth_average

        s = tf.nn.softmax(weight_each_vertex)

        return s
