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
from tensorflow_transform.beam.tft_beam_io import beam_metadata_io
import tensorflow_model_analysis as tfma
import tensorflow_hub as hub
import apache_beam as beam
import shutil
import os
import config
import variables
print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)

def build_estimator(model_dir, model_type, embedding_type, learning_rate,
                    hidden_units, dropout,
                    l1_regularization_strength, l2_regularization_strength):
  
    if embedding_type == 'nnlm':
        module_url = 'https://tfhub.dev/google/nnlm-en-dim128/1'
        embedding_size = 128
    elif embedding_type == 'universal-sentence-encoder':
        module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
        embedding_size = 512
    elif embedding_type == 'elmo':
        module_url = 'https://tfhub.dev/google/elmo/2'
        embedding_size = 1024
    else:
        raise InputError('Embedding type must be one of "nnlm", "universal-sentence-encoder", "elmo"')
    
    if model_type in ('linear', 'dnn-linear-combined'):
        bow_indices = tf.feature_column.categorical_column_with_identity('bow_indices', num_buckets=config.MAX_TOKENS+1)
        weighted_bow = tf.feature_column.weighted_categorical_column(bow_indices, 'bow_weight')
    if model_type in ('dnn', 'dnn-linear-comnbined'):
        embedding = hub.text_embedding_column(config.TOKENIZE_COL, module_url, trainable=False)
    if model_type in ('cnn', 'rnn'):
        # have to create custom module spec to embed each word without
        # combining into single sentence embedding
        def _embed():
            text_feature = tf.placeholder(tf.string, shape=(None,))
            words = tf.string_split(text_feature)
            batch_size = words.dense_shape[0]
            dense_words = tf.sparse_to_dense(
                sparse_indices=words.indices,
                sparse_values=words.values,
                default_value='',
                output_shape=(batch_size, config.MAX_SEQ_LEN)
            )
            embed = hub.Module(module_url, trainable=False)
            embeddings = tf.map_fn(lambda token: embed(token), dense_words, dtype=tf.float32, name='text_seq_embedding')
            hub.add_signature(inputs=text_feature, outputs=embeddings)
            
        embedding_spec = hub.create_module_spec(_embed)
        
        # embedding shape: (batch_size, config.MAX_SEQ_LEN, embedding_size)
        # embedding = hub.text_embedding_column(config.TOKENIZE_COL, embedding_spec)
        embedding = hub.feature_column._TextEmbeddingColumn(config.TOKENIZE_COL, embedding_spec, trainable=False)
    
    if model_type == 'linear':
        feature_columns = [weighted_bow]
        
        estimator = tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
            n_classes=variables.N_CLASSES,
            label_vocabulary=variables.LABEL_VOCABULARY,
            model_dir=model_dir,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=learning_rate,
                l1_regularization_strength=l1_regularization_strength,
                l2_regularization_strength=l2_regularization_strength
            )
        )
    elif model_type == 'dnn':
        feature_columns = [embedding]
        
        estimator = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            n_classes=variables.N_CLASSES,
            label_vocabulary=variables.LABEL_VOCABULARY,
            model_dir=model_dir,
            optimizer=tf.train.AdamOptimizer(
                learning_rate=learning_rate,
            ),
            dropout=dropout
        )
    elif model_type == 'dnn-linear-combined':
        dnn_features = [embedding]
        linear_features = [weighted_bow]
        
        estimator = tf.estimator.DNNLinearCombinedClassifier(
            linear_feature_columns=linear_features,
            linear_optimizer=tf.train.FtrlOptimizer(
                learning_rate=learning_rate,
                l1_regularization_strength=l1_regularization_strength,
                l2_regularization_strength=l2_regularization_strength
            ),
            dnn_feature_columns=dnn_features,
            dnn_optimizer=tf.train.AdamOptimizer(
                learning_rate=learning_rate,
            ),
            dnn_dropout=dropout,
            dnn_hidden_units=hidden_units,
            n_classes=variables.N_CLASSES,
            label_vocabulary=variables.LABEL_VOCABULARY,
            model_dir=model_dir,
            batch_norm=True
        )
    elif model_type == 'rnn':      
        text_input = tf.keras.layers.Input(shape=(config.MAX_SEQ_LEN, embedding_size), name='text_seq_embedding', dtype=tf.float32)
        processed = text_input
        for unit in hidden_units:
            processed = tf.keras.layers.LSTM(unit)(processed)
        processed = tf.keras.layers.Dense(128, activation='relu')(processed)
        processed = tf.keras.layers.Dropout(dropout)(processed)
        output = tf.keras.layers.Dense(variables.N_CLASSES, activation='sigmoid', name='probabilities')(processed)
        
        model = tf.keras.Model(inputs=text_input, outputs=output)
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        estimator = tf.keras.estimator.model_to_estimator(model)
        
    else:
        raise InputErorr('Model type must be one of "linear" or "dnn"')
        
    if len(config.PASSTHROUGH_COLS) > 0:
        estimator = tf.contrib.estimator.forward_features(estimator, config.PASSTHROUGH_COLS)
        
    def get_model_fn_with_removed_outputs(estimator):
        def _model_fn(features, labels, mode):
            config = estimator.config
            model_fn_ops = estimator._model_fn(features=features, labels=labels, mode=mode, config=config)
            model_fn_ops.predictions['probability'] = tf.reduce_max(model_fn_ops.predictions['probabilities'], axis=-1)
            for key in ('logits', 'logistic', 'probabilities', 'class_ids'):
                try:
                    model_fn_ops.predictions.pop(key)
                except KeyError:
                    pass
            return model_fn_ops
        return _model_fn
        
    estimator = tf.estimator.Estimator(model_fn=get_model_fn_with_removed_outputs(estimator))
    
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
        exclude_raw_keys=[config.LABEL_COL]
    )

def make_serving_input_fn(args):
    transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))
    
    def _input_fn():
        feature_placeholders = {
            column_name: tf.placeholder(tf.string, [None]) for column_name in config.STRING_COLS
        }
        feature_placeholders.update({
            column_name: tf.placeholder(tf.float32, [None]) for column_name in config.NUMERIC_COLS
        })
        feature_placeholders.pop(config.LABEL_COL)
        
        _, features = saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir,
            feature_placeholders
        )
        
        # so that outputs are consistently in lists
        if len(config.PASSTHROUGH_COLS) > 0:
            for col in config.PASSTHROUGH_COLS:
                features[col] = tf.expand_dims(tf.identity(feature_placeholders[col]), axis=1)
        
        return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
    
    return _input_fn


def make_eval_input_fn(args):
    transform_savedmodel_dir = (
        os.path.join(args['metadata_path'], 'transform_fn'))
    
    def _input_fn():
        metadata = beam_metadata_io.metadata_io.read_metadata('data/tft/metadata/rawdata_metadata/')
        raw_feature_spec = metadata.schema.as_feature_spec()

        serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_example_tensor')

        features = tf.parse_example(serialized_tf_example, raw_feature_spec)
        
        _, transformed_features = saved_transform_io.partially_apply_saved_transform(
            transform_savedmodel_dir,
            features
        )
        
        receiver_tensors = {'examples': serialized_tf_example}
        
        return tfma.export.EvalInputReceiver(
            features=transformed_features,
            receiver_tensors=receiver_tensors,
            labels=transformed_features[config.LABEL_COL]
        )
    
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
        label_keys=[config.LABEL_COL + '_ one_hot' if args['model_type'] in ('rnn', 'cnn') else config.LABEL_COL],
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
    
    if args['debug'] == True:
        train_spec = tf.estimator.TrainSpec(
            input_fn=read_dataset(args, tf.estimator.ModeKeys.TRAIN),
            max_steps=10
        )
        
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        
        for result in estimator.predict(input_fn=read_dataset(args, mode=tf.estimator.ModeKeys.EVAL)):
            print(result)
    else:
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
        tfma.export.export_eval_savedmodel(
            estimator=estimator,
            export_dir_base='model_trained/eval/tfma/',
            eval_input_receiver_fn=make_eval_input_fn(args)
        )
    
    # export results
#     if not os.path.exists('data/output'):
#         os.mkdir('data/output')
#     eval_preds = pd.DataFrame(list(estimator.predict(input_fn=read_dataset(args, tf.estimator.ModeKeys.EVAL))))
#     probabilities = list(list(arr) for arr in eval_preds['probabilities']) # pandas is weird with how it stores arrays
#     with tf.Session() as sess:
#         eval_preds['probability'] = sess.run(tf.reduce_max(probabilities, reduction_indices=[1]))
#     eval_preds['pred_' + config.LABEL_COL] = eval_preds['classes'].map(lambda x: x[0]) # predictions come in a list per row
#     eval_preds = eval_preds[['pred_' + config.LABEL_COL, 'probability']]
#     raw_eval_df = pd.concat([
#         pd.read_csv(f, sep=config.DELIM, names=config.RENAMED_COLS)
#         for f in tf.gfile.Glob('data/split/eval*.csv')], 
#         axis=0, ignore_index=True)
#     cols = list(raw_eval_df.columns)
#     cols.remove(config.LABEL_COL)
#     raw_eval_df = raw_eval_df[cols + [config.LABEL_COL]]
#     for col in ['pred_' + config.LABEL_COL, 'probability']:
#         raw_eval_df[col] = eval_preds[col]
#     raw_eval_df['wrong'] = (raw_eval_df['pred_' + config.LABEL_COL] != raw_eval_df[config.LABEL_COL]).astype(int)
#     raw_eval_df.to_excel('data/output/eval_with_preds.xlsx', index=False)
    
    
def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))