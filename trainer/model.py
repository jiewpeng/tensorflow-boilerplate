#!/usr/bin/env python


from __future__ import print_function, division, absolute_import # python 2 compatibility
import sys
reload(sys)
sys.setdefaultencoding('utf8')
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
from config import REGION, BUCKET, PROJECT, LABEL_COL, PASSTHROUGH_COLS, STRING_COLS, NUMERIC_COLS, DELIM, RENAMED_COLS
print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)

with open('data/misc/labels.txt', 'r') as f:
    LABEL_VOCABULARY = f.readline().split(DELIM)
    N_CLASSES = len(LABEL_VOCABULARY)


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
        
    if len(PASSTHROUGH_COLS) > 0:
        estimator = tf.contrib.estimator.forward_features(estimator, PASSTHROUGH_COLS)

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
        exclude_raw_keys=[LABEL_COL]
    )

def make_serving_input_fn(args):
    transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))
    
    def _input_fn():
        feature_placeholders = {
            column_name: tf.placeholder(tf.string, [None]) for column_name in STRING_COLS
        }
        feature_placeholders.update({
            column_name: tf.placeholder(tf.float32, [None]) for column_name in NUMERIC_COLS
        })
        feature_placeholders.pop(LABEL_COL)
        
        _, features = saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir,
            feature_placeholders
        )
        
        # so that outputs are consistently in lists
        if len(PASSTHROUGH_COLS) > 0:
            for col in PASSTHROUGH_COLS:
                features[col] = tf.expand_dims(tf.identity(feature_placeholders[col]), axis=1)
        
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
        label_keys=[LABEL_COL],
        reader=gzip_reader_fn,
        randomize_input=(mode == tf.estimator.ModeKeys.TRAIN),
        num_epochs=(None if mode == tf.estimator.ModeKeys.TRAIN else 1)
    )


# create tf.estimator train and evaluate function
def train_and_evaluate(args):
    # figure out train steps based on no. of epochs, no. of rows in dataset and batch size
    tfrecord_options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    nrows = sum(
        sum(1 for _ in tf.python_io.tf_record_iterator(f, options=tfrecord_options)) 
        for f in tf.gfile.Glob(args['train_data_paths'])
    )
    num_epochs = args['num_epochs']
    batch_size = args['train_batch_size']
    if batch_size > nrows:
        batch_size = nrows
    max_steps = num_epochs * nrows / batch_size
    
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
        max_steps=max_steps
    )
    
    exporter = tf.estimator.LatestExporter('exporter', make_serving_input_fn(args))
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(args, tf.estimator.ModeKeys.EVAL),
        steps=None,
        exporters=exporter
    )
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    # export results
    if not os.path.exists('data/output'):
        os.mkdir('data/output')
    eval_preds = pd.DataFrame(list(estimator.predict(input_fn=read_dataset(args, tf.estimator.ModeKeys.EVAL))))
    probabilities = list(list(arr) for arr in eval_preds['probabilities']) # pandas is weird with how it stores arrays
    with tf.Session() as sess:
        eval_preds['probability'] = sess.run(tf.reduce_max(probabilities, reduction_indices=[1]))
    eval_preds['pred_' + LABEL_COL] = eval_preds['classes'].map(lambda x: x[0]) # predictions come in a list per row
    eval_preds = eval_preds[['pred_' + LABEL_COL, 'probability']]
    raw_eval_df = pd.concat([
        pd.read_csv(f, sep=DELIM, names=RENAMED_COLS)
        for f in tf.gfile.Glob('data/split/eval*.csv')], 
        axis=0, ignore_index=True)
    cols = list(raw_eval_df.columns)
    cols.remove(LABEL_COL)
    raw_eval_df = raw_eval_df[cols + [LABEL_COL]]
    for col in ['pred_' + LABEL_COL, 'probability']:
        raw_eval_df[col] = eval_preds[col]
    raw_eval_df['wrong'] = (raw_eval_df['pred_' + LABEL_COL] != raw_eval_df[LABEL_COL]).astype(int)
    raw_eval_df.to_excel('data/output/eval_with_preds.xlsx', index=False)
    
    
def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_eval_metrics():
    return {
        'accuracy': tflearn.MetricSpec(metric_fn=metrics.streaming_accuracy),
        'training/hptuning/metric': tflearn.MetricSpec(metric_fn=metrics.streaming_accuracy),
    }