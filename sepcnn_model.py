"""Module to create sepCNN model

Helper functions to create a separable CNN model. The main model function is
meant to be a tf.estimator.Estimator, and takes in hyper-parameters through
the `params` argument
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.contrib.lookup import index_to_string_table_from_file
from tensorflow.contrib.lookup import index_table_from_file
import tensorflow_hub as hub

from config import MAX_TOKENS


class CustomSeparableConv1D(SeparableConv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CustomSeparableConv1D, self).__init__(
            filters,
            kernel_size,
            padding='same',
            activation='selu',
            bias_initializer='glorot_uniform',
            depthwise_initializer='glorot_uniform',
            **kwargs
        )


def create_model(blocks,
                 filters,
                 kernel_size,
                 pool_size,
                 dropout_rate,
                 embedding_size,
                 max_seq_len,
                 embedding_trainable,
                 op_units):
    
    description_input = Input(dtype=tf.int32, shape=(max_seq_len,))
    vendor_name_input = Input(dtype=tf.float32, shape=(5,))
    
    description = Embedding(
        input_dim=MAX_TOKENS,
        output_dim=embedding_size,
        input_length=max_seq_len,
        weights=[embedding_matrix],
        trainable=embedding_trainable,
        name='description_embedding')(description_input)
    vendor_name = Embedding(
        input_dim=MAX_TOKENS,
        output_dim=embedding_size,
        input_length=5,
        weights=[embedding_matrix],
        trainable=embedding_trainable,
        name='vendor_name_embedding')(vendor_name_input)

    for block in range(blocks - 1):
        description = Dropout(dropout_rate)(description)
        description = CustomSeparableConv1D(filters, kernel_size)(description)
        tf.summary.scalar('description_fraction_zero_weights_sepconv_block{}_1'.format(block + 1), tf.math.zero_fraction(description))
        tf.summary.histogram('description_sepconv_block{}_1'.format(block + 1), tf.identity(description))
        description = CustomSeparableConv1D(filters, kernel_size)(description)
        tf.summary.scalar('description_fraction_zero_weights_sepconv_block{}_2'.format(block + 1), tf.math.zero_fraction(tf.identity(description)))
        tf.summary.histogram('description_sepconv_block{}_2'.format(block + 1), tf.identity(description))
        description = MaxPooling1D(pool_size)(description)

    description = CustomSeparableConv1D(filters * 2, kernel_size)(description)
    tf.summary.scalar('description_fraction_zero_weights_sepconv_block{}_1'.format(blocks), tf.math.zero_fraction(tf.identity(description)))
    tf.summary.histogram('description_sepconv_block{}_1'.format(blocks), tf.identity(description))
    description = CustomSeparableConv1D(filters * 2, kernel_size)(description)
    tf.summary.scalar('description_fraction_zero_weights_sepconv_block{}_2'.format(blocks), tf.math.zero_fraction(tf.identity(description)))
    tf.summary.histogram('description_sepconv_block{}_1'.format(blocks), tf.identity(description))
    description = GlobalAveragePooling1D()(description)
    description = Dropout(dropout_rate)(description)
    
    vendor_name = CustomSeparableConv1D(filters, kernel_size)(vendor_name)
    tf.summary.scalar('vendorname_fraction_zero_weights_sepconv_block{}_1'.format(blocks), tf.math.zero_fraction(tf.identity(description)))
    tf.summary.histogram('vendorname_sepconv_block{}_1'.format(blocks), tf.identity(description))
    vendor_name = GlobalAveragePooling1D()(vendor_name)
    vendor_name = Dropout(dropout_rate)(vendor_name)
    
    concatenated = Concatenate()([description, vendor_name])

    logits = Dense(op_units, activation='softmax')(concatenated)

    model = Model(inputs=[description_input, vendor_name_input], outputs=logits)
    
    return model


def pad_seq(text, max_seq_len):
    reshaped_input = tf.reshape(text, [-1])
    split = tf.strings.split(reshaped_input)
    split = tf.sparse.to_dense(split, default_value='')
    seq_len = tf.shape(split)[1]
    batch_size = tf.shape(split)[0]
    split = tf.cond(
        seq_len < max_seq_len,
        lambda: tf.pad(split, [[0, 0], [0, max_seq_len - seq_len]], constant_values=''),
        lambda: tf.slice(split, [0, 0], [batch_size, max_seq_len])
    )
    return split


def sepcnn_model_fn(features, labels, mode, params):
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
            embedding_trainable: bool, true if embedding layer is trainable.
            embedding_size: int, dimension of embedding

    # Returns
        A tf.estimator.EstimatorSpec of a sepCNN model.
    """
    index_to_string_table = index_to_string_table_from_file('data/misc/labels.txt')
    string_to_index_table = index_table_from_file('data/misc/labels.txt')
    
    num_classes = params.get('num_classes', 2)
    embedding_type = params.get('embedding_type', 'nnlm')
    embedding_trainable = params.get('embedding_trainable', False)
    blocks = params.get('blocks', 2)
    filters = params.get('filters', 32)
    kernel_size = params.get('kernel_size', 3)
    pool_size = params.get('pool_size', 3)
    max_seq_len = params.get('max_seq_len', 200)
    dropout_rate = params.get('dropout_rate', 0.2)
    learning_rate = params.get('learning_rate', 1e-4)

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    embedding_info = {
        'nnlm': {'url': 'https://tfhub.dev/google/nnlm-en-dim128/1', 'embedding_size': 128},
        'wikiwords250': {'url': 'https://tfhub.dev/google/Wiki-words-250/1', 'embedding_size': 250},
        'elmo': {'url': 'https://tfhub.dev/google/elmo/2', 'embedding_size': 1024},
        'universal-sentence-encoder': {'url': 'https://tfhub.dev/google/universal-sentence-encoder/2', 'embedding_size': 512}
    }
    embedding_url = embedding_info[embedding_type]['url']
    embedding_size = embedding_info[embedding_type]['embedding_size']

    model = create_model(
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dropout_rate=dropout_rate,
        max_seq_len=max_seq_len,
        embedding_size=embedding_size,
        op_units=op_units
    )

    embed = hub.Module(embedding_url, trainable=embedding_trainable)

    description = features['full_description']
    description = tf.reshape(description, [-1])
    vendor_name = features['vendor_name']
    vendor_name = tf.reshape(vendor_name, [-1])

    description_seq = pad_seq(description, max_seq_len)
    vendor_name_seq = pad_seq(vendor_name, 5)
    description_embedding = tf.map_fn(embed, description_seq, dtype=tf.float32, name='description_embedding')
    vendor_name_embedding = tf.map_fn(embed, vendor_name_seq, dtype=tf.float32, name='vendor_name_embedding')

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model([description_embedding, vendor_name_embedding], training=False)
        
        predictions = {
            'classes': index_to_string_table.lookup(tf.argmax(logits, axis=1)),
            'probabilities': tf.nn.softmax(logits),
        }
        
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = model([description_embedding, vendor_name_embedding], training=True)
        label_ids = string_to_index_table.lookup(labels)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_ids, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model([description_embedding, vendor_name_embedding], training=False)
        label_ids = string_to_index_table.lookup(labels)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_ids, logits=logits)
        
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=label_ids, predictions=tf.argmax(logits, axis=1)),
            })


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
