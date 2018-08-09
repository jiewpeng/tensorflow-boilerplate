#!/usr/bin/env python


from __future__ import print_function, division, absolute_import # python 2 compatibility
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.metrics as metrics
from tensorflow_transform.saved import input_fn_maker, saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io
import tensorflow_hub as hub
import apache_beam as beam
import shutil
import os
print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ['spam', 'text']
LABEL_COLUMN = 'spam'
LABEL_VOCABULARY = ['ham', 'spam']
N_CLASSES = len(LABEL_VOCABULARY)
DEFAULTS = [['spam'], ["FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 1.50 to rcv"]]
INPUT_COLUMNS = [
    tf.placeholder(tf.string, name='text')
]
PASSTHROUGH_COLUMNS = ['key']


def build_estimator(model_dir, model_type, embedding_type, learning_rate,
                    hidden_units, dropout,
                    l1_regularization_strength, l2_regularization_strength):
  
    if embedding_type == 'nnlm':
        module_url = 'https://tfhub.dev/google/nnlm-en-dim128/1'
    elif embedding_type == 'universal-sentence-encoder':
        module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
    elif embedding_type == 'elmo':
        module_url = 'https://tfhub.dev/google/elmo/2'
    elif embedding_type is None:
        pass
    else:
        raise InputError('Embedding type must be one of "nnlm", "universal-sentence-encoder", "elmo", None')
    
    if embedding_type is not None:
        embedding = hub.text_embedding_column('text', module_url, trainable=False)
        
    feature_columns = embedding = [embedding]
    
    if model_type == 'linear':
        estimator = tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
            n_classes=N_CLASSES,
            label_vocabulary=LABEL_VOCABULARY,
            model_dir=model_dir,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=learning_rate,
                l1_regularization_strength=l1_regularization_strength,
                l2_regularization_strength=l2_regularization_strength
            )
        )
    elif model_type == 'dnn':
        estimator = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            n_classes=N_CLASSES,
            label_vocabulary=LABEL_VOCABULARY,
            model_dir=model_dir,
            optimizer=tf.train.AdamOptimizer(
                learning_rate=learning_rate,
            ),
            dropout=dropout
        )
    else:
        raise InputErorr('Model type must be one of "linear" or "dnn"')
        
    if len(PASSTHROUGH_COLUMNS) > 0:
        estimator = tf.contrib.estimator.forward_features(estimator, PASSTHROUGH_COLUMNS)

    return estimator
        
# Serving input function
def make_serving_input_fn_for_base64_json(args):
    raw_metadata = metadata_io.read_metadata(
        os.path.join(args['metadata_path'], 'rawdata_metadata'))
    transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))
    return input_fn_maker.build_parsing_transforming_serving_input_receiver_fn(
        raw_metadata,
        transform_savedmodel_dir,
        exclude_raw_keys=[LABEL_COLUMN]
    )

def make_serving_input_fn(args):
    transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))
    
    def _input_fn():
        feature_placeholders = {
            column_name: tf.placeholder(tf.string, [None]) for column_name in 'text'.split(',')
        }
        
        if len(PASSTHROUGH_COLUMNS) > 0:
            for col in PASSTHROUGH_COLUMNS:
                feature_placeholders[col] = tf.placeholder(tf.string, [None])
        
        _, features = saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir,
            feature_placeholders
        )
        
        if len(PASSTHROUGH_COLUMNS) > 0:
            for col in PASSTHROUGH_COLUMNS:
                features[col] = tf.identity(feature_placeholders[col])
        
        return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
    
    return _input_fn


# training, eval and test input function
def read_dataset(args, mode):
    batch_size = args['train_batch_size']
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_paths = args['train_data_paths']
    elif mode == tf.estimator.ModeKeys.EVAL:
        input_paths = args['eval_data_paths']
    else:
        input_paths = args['test_data_paths']
    
    transformed_metadata = metadata_io.read_metadata(
        os.path.join(args['metadata_path'], 'transformed_metadata'))
    
    return input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern = (input_paths[0] if len(input_paths) == 1 else input_paths),
        training_batch_size=batch_size,
        label_keys=[LABEL_COLUMN],
        reader=gzip_reader_fn,
        randomize_input=(mode == tf.estimator.ModeKeys.TRAIN),
        num_epochs=(None if mode == tf.estimator.ModeKeys.TRAIN else 1)
    )


# create tf.estimator train and evaluate function
def train_and_evaluate(args):
    # modify according to build_estimator function
    estimator = build_estimator(
        args['model_dir'],
        args['model_type'],
        args['embedding_type'],
        args['learning_rate'],
        args['hidden_units'].split(' '),
        args['dropout'],
        args['l1_regularization_strength'],
        args['l2_regularization_strength']
    )
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(args, tf.estimator.ModeKeys.TRAIN),
        max_steps=args['train_steps']
    )
    
    exporter = tf.estimator.LatestExporter('exporter', make_serving_input_fn(args))
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(args, tf.estimator.ModeKeys.EVAL),
        steps=None,
        exporters=exporter
    )
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    
def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_eval_metrics():
    return {
        'accuracy': tflearn.MetricSpec(metric_fn=metrics.streaming_accuracy),
        'training/hptuning/metric': tflearn.MetricSpec(metric_fn=metrics.streaming_accuracy),
    }