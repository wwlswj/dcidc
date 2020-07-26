# -*- coding: utf-8 -*-

import tensorflow as tf
import data_parse as dp
import network_specs as specs


def fc_layers(input_data, layer_dimensions, layer_activations, layer_names):
    for dimension, activation, name in zip(layer_dimensions, layer_activations, layer_names):
        input_data = tf.layers.dense(inputs=input_data, units=dimension, activation=activation,
                                     name=name, reuse=tf.AUTO_REUSE)
    return input_data


def err_func(x, y):
    return tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1)
