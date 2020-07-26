# -*- coding: utf-8 -*-

import tensorflow as tf

activations = {'relu': tf.nn.relu,
                        'tanh': tf.nn.tanh,
                        'sigmoid': tf.nn.sigmoid,
                        'relu6': tf.nn.relu6,
                        'crelu': tf.nn.crelu,
                        'softplus': tf.nn.softplus,
                        'elu': tf.nn.elu,
                        'selu': tf.nn.selu}


def make_specs(input_dimension, embedding_dimension, hidden_dimensions=[500, 500, 2000], activation_name='relu'):
    layer_dimensions_front = hidden_dimensions
    layer_dimensions_tail = layer_dimensions_front[::-1]
    layer_dimensions_front.append(embedding_dimension)
    layer_dimensions_tail.append(input_dimension)
    layer_dimensions = layer_dimensions_front + layer_dimensions_tail

    layer_activation = activations[str.lower(activation_name)]
    layer_activations = [layer_activation for _ in range(len(layer_dimensions))]
    layer_activations[int(len(layer_dimensions)/2-1)] = None
    layer_activations[-1] = None

    layer_names_front = []
    layer_names_tail = []
    for i in range(len(layer_dimensions_front)):
        number = i+1
        if len(layer_dimensions_front) == number:
            encoder_name = 'embedding'
            decoder_name = 'output'
        else:
            encoder_name = 'encoder_hidden_' + str(number)
            decoder_name = 'decoder_hidden_' + str(number)
        layer_names_front.append(encoder_name)
        layer_names_tail.append(decoder_name)
    layer_names = layer_names_front + layer_names_tail

    return layer_dimensions, layer_activations, layer_names


def make_specs_nn(output_dimension, hidden_dimensions=[200, 300, 500, 300, 200], activation_name='tanh'):
    layer_dimensions = list(hidden_dimensions)
    layer_dimensions.append(output_dimension)
    layer_activation = activations[activation_name.lower()]
    layer_activations = [layer_activation for _ in range(len(layer_dimensions))]
    layer_activations[-1] = None
    layer_names = ['hidden_layer_' + str(i+1) for i in range(len(layer_dimensions))]
    layer_names[-1] = 'output'
    return layer_dimensions, layer_activations, layer_names
