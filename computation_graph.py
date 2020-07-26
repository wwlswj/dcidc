# -*- coding: utf-8 -*-

import tensorflow as tf
import data_parse as dp
import network_specs as specs


def fc_layers(input_data, layer_dimensions, layer_activations, layer_names):
    for dimension, activation, name in zip(layer_dimensions, layer_activations, layer_names):
        input_data = tf.layers.dense(inputs=input_data, units=dimension, activation=activation,
                                     name=name, reuse=tf.AUTO_REUSE)
    return input_data


def auto_encoder(input_data, layer_dimensions, layer_activations, layer_names):
    mid_ind = int(len(layer_dimensions) / 2)

    # Encoder
    embedding = fc_layers(input_data, layer_dimensions[:mid_ind], layer_activations[:mid_ind], layer_names[:mid_ind])
    # Decoder
    output = fc_layers(embedding, layer_dimensions[mid_ind:], layer_activations[mid_ind:], layer_names[mid_ind:])

    return embedding, output


def err_func(x, y):
    return tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1)


class ComputationGraph(object):
    def __init__(self, file_name_key, alpha, embedding_dimension=None,
                 hidden_dimensions=[500, 500, 2000], activation_name='relu'):

        self.data, self.labels, self.label_values, self.label_colors, self.image_shape, self.names = \
            dp.load_standard_mat(file_name_key)
        self.input_dimension = self.data.shape[1]
        self.embedding_dimension = embedding_dimension
        if not self.embedding_dimension:
            self.embedding_dimension = len(self.label_values)
        self.output_dimension = self.input_dimension
        self.n_cluster = len(self.label_values)
        self.n_sample = self.data.shape[0]

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.layer_dimensions, self.layer_activations, self.layer_names = \
                specs.make_specs(self.input_dimension, self.embedding_dimension, hidden_dimensions, activation_name)

            # The placeholder tensor for input data
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dimension])

            # Get the embedding and output of the auto-encoder
            self.embedding, self.output = auto_encoder(self.input, self.layer_dimensions,
                                                       self.layer_activations, self.layer_names)

            # Get the loss of the auto-encoder network
            self.ae_loss = tf.reduce_mean(err_func(self.input, self.output))

            # Clustering loss computation
            # Tensor for cluster representatives
            self.cluster_rep = tf.Variable(tf.random_uniform([self.n_cluster, self.embedding_dimension],
                                                             minval=0, maxval=1, dtype=tf.float32),
                                           name='cluster_rep', dtype=tf.float32, trainable=True)

            # Inter-class loss computation
            f_2_ones = tf.ones(shape=(self.n_cluster, self.n_cluster), dtype=tf.float32)
            f_2_eye = tf.eye(num_rows=self.n_cluster, num_columns=self.n_cluster, dtype=tf.float32)
            f_2 = tf.subtract(f_2_ones, f_2_eye)
            f_2_weight = tf.multiply(1.0 / (self.n_cluster - 1.0), f_2)
            cluster_rep_sum = tf.matmul(f_2_weight, self.cluster_rep)
            self.inter_class_loss = tf.reduce_mean(err_func(self.cluster_rep, cluster_rep_sum))

            # Clustering assignments for all samples in the dataset
            initial_clustering_assign = tf.random_uniform(minval=0, maxval=self.n_cluster,
                                                          dtype=tf.int32, shape=[self.n_sample])
            self.cluster_assign = tf.Variable(initial_clustering_assign, name='cluster_assign',
                                              dtype=tf.int32, trainable=False)

            # Get the cluster representatives corresponding to each sample
            self.corresponded_rep = tf.gather(self.cluster_rep, self.cluster_assign)

            # Calculating the intra-class errors
            self.intra_class_loss = tf.reduce_mean(err_func(self.embedding, self.corresponded_rep))

            # Get the total loss
            self.loss = tf.add(self.ae_loss, tf.multiply(alpha, tf.div(self.intra_class_loss, self.inter_class_loss)))

            # The optimizer is defined to minimize these losses
            optimizer = tf.train.AdamOptimizer()
            self.pre_train_op = optimizer.minimize(self.ae_loss)
            self.train_op = optimizer.minimize(self.loss)

            # Update the clustering assignments
            embedding_expand = tf.expand_dims(self.embedding, axis=1)
            cluster_rep_tiled_reshaped = tf.reshape(tf.tile(self.cluster_rep, multiples=(self.n_sample, 1)),
                                                    shape=(self.n_sample, self.n_cluster, self.embedding_dimension))
            distance_op = tf.reduce_sum(tf.square(tf.subtract(embedding_expand, cluster_rep_tiled_reshaped)), axis=2)
            new_assign = tf.cast(tf.argmin(distance_op, axis=1), dtype=tf.int32)
            self.cluster_assign_update = tf.assign(self.cluster_assign, new_assign)

            # As an option, one can manually update the cluster representatives, however, this is
            # usually done automatically by the network.
            self.cluster_counts = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[self.n_cluster]),
                                              trainable=False, dtype=tf.float32, name='cluster_counts')
            cluster_eye = tf.eye(num_rows=self.n_cluster, num_columns=self.n_cluster, dtype=tf.float32)
            cluster_eye_corresponded = tf.gather(cluster_eye, self.cluster_assign)
            self.cluster_counts_update = tf.assign(self.cluster_counts, tf.reduce_sum(cluster_eye_corresponded, axis=0))

            f_1 = tf.transpose(cluster_eye_corresponded)
            cluster_rep_new = tf.div(tf.matmul(f_1, self.embedding), tf.expand_dims(self.cluster_counts, axis=1))
            self.cluster_rep_update = tf.assign(self.cluster_rep, cluster_rep_new)

