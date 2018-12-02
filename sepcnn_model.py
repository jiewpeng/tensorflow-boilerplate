"""Module to create sepCNN model

Helper functions to create a separable CNN model. The main model function is
meant to be a tf.estimator.Estimator, and takes in hyper-parameters through
the `params` argument
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.layers import separable_conv1d
from tensorflow.layers import max_pooling1d
from tensorflow.layers import dense
from tensorflow.layers import dropout
from tensorflow.layers import batch_normalization
from tensorflow.contrib.lookup import index_to_string_table_from_tensor
from tensorflow.contrib.lookup import index_table_from_tensor
import tensorflow_hub as hub

from config import MAX_SEQ_LEN
from config import TOKENIZE_COL
from variables import LABEL_VOCABULARY


def cnn_model_fn(features, labels, mode, params):
    """Creates an instance of a separable CNN model.

    # Arguments
        features: dict of feature tensors
        labels: tensor of labels
        mode: tf.estimator.ModeKey, or str specifying whether the model should
            be training, evaluating or predicting
        params: dict with the following keys
            blocks: int, number of pairs of sepCNN and pooling blocks in the model.
            filters: int, output dimension of the layers.
            kernel_size: int, length of the convolution window.
            dropout_rate: float, percentage of input to drop at Dropout layers.
            pool_size: int, factor by which to downscale input at MaxPooling layer.
            num_classes: int, number of output classes.
            module_url: str, a URL to a TFHub text embedding
            is_embedding_trainable: bool, true if embedding layer is trainable.
            embedding_size: int, dimension of embedding

    # Returns
        A tf.estimator.EstimatorSpec of a sepCNN model.
    """
    index_to_string_table = index_to_string_table_from_tensor(LABEL_VOCABULARY)
    string_to_index_table = index_table_from_tensor(LABEL_VOCABULARY)
    num_classes = params['num_classes']
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    embed = hub.Module(
        params['module_url'],
        trainable=params['is_embedding_trainable'])

    # when predicting, tensors may come expanded or not - reshape to ensure
    # that all input tensors are the same shape as when training/evaluating
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features[TOKENIZE_COL]
        input_layer = tf.reshape(input_layer, (tf.shape(input_layer)[0], 1))
    else:
        input_layer = features[TOKENIZE_COL]

    # Input shape is (batch_size, 1), need to flatten before passing into
    # string split.
    input_layer = tf.reshape(input_layer, [-1])

    # split string before passing into embedding to get out a sequence of
    # embeddings instead of a combined/averaged embedding, which is the default
    # behaviour of the TFHub embeddings
    split = tf.strings.split(input_layer)
    split = tf.sparse.to_dense(split, default_value='')
    seq_len = tf.shape(split)[1]    
    batch_size = tf.shape(split)[0]
    # shorten or pad sequence to maintain a constant sequence length
    split = tf.cond(
        seq_len < MAX_SEQ_LEN,
        lambda: tf.pad(split, [[0, 0], [0, MAX_SEQ_LEN - seq_len]], constant_values=''),
        lambda: tf.slice(split, [0, 0], [batch_size, MAX_SEQ_LEN])
    )

    embeddings = tf.map_fn(embed, split, dtype=tf.float32)
    output = embeddings

    # separabale conv blocks
    for _ in range(params['blocks'] - 1):
        output = dropout(output, rate=params['dropout_rate'])
        output = separable_conv1d(
            output,
            filters=params['filters'],
            kernel_size=params['kernel_size'],
            padding='same',
            activation='relu',
            bias_initializer='random_uniform',
            depthwise_initializer='random_uniform'
        )
        output = separable_conv1d(
            output,
            filters=params['filters'],
            kernel_size=params['kernel_size'],
            padding='same',
            activation='relu',
            bias_initializer='random_uniform',
            depthwise_initializer='random_uniform'
        )
        output = max_pooling1d(
            output, 
            pool_size=params['pool_size'], 
            strides=params['pool_size'])
        output = batch_normalization(output)

    output = separable_conv1d(
        output,
        filters=params['filters'],
        kernel_size=params['kernel_size'],
        padding='same',
        activation='relu',
        bias_initializer='random_uniform',
        depthwise_initializer='random_uniform'
    )
    output = separable_conv1d(
        output,
        filters=params['filters'],
        kernel_size=params['kernel_size'],
        padding='same',
        activation='relu',
        bias_initializer='random_uniform',
        depthwise_initializer='random_uniform'
    )

    output = tf.reduce_mean(output, axis=1)  # global average pooling 1d
    output = batch_normalization(output)
    output = dropout(output, rate=params['dropout_rate'])

    logits = dense(output, units=op_units, activation=op_activation)

    if op_activation == 'sigmoid':
        class_ids = tf.cast(logits > 0.5, tf.int64)
    else:
        class_ids = tf.argmax(logits, axis=1)

    # get back original labels
    classes = index_to_string_table.lookup(class_ids)

    predictions = {
        'prediction': classes,
        'probabilities': logits
    }

    # return early if predicting, so we won't try to lookup the labels or 
    # calculate loss when the input doesn't have any labels
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    label_ids = string_to_index_table.lookup(labels)
    label_ids = tf.reshape(label_ids, (tf.shape(label_ids)[0], 1))

    if op_activation == 'sigmoid':
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_ids, logits=logits)
    else:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_ids, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
    else:  # eval
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=label_ids, predictions=class_ids)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.
    # Arguments
        num_classes: int, number of classes.
    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation
