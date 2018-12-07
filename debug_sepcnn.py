from trainer.model import build_estimator, read_dataset
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import shutil


estimator = build_estimator(
    model_dir='./model_trained', 
    model_type='sepcnn', 
    embedding_type='nnlm', 
    learning_rate=0.01,
    hidden_units=[128, 64],
    dropout=0.5,
    l1_regularization_strength=0.01, 
    l2_regularization_strength=0.01,
    blocks=1, 
    filters=8, 
    kernel_size=3, 
    pool_size=3)

args = {
    'train_data_paths': './data/tft/train*',
    'eval_data_paths': './data/tft/eval*',
    'model_dir': './model_trained',
    'num_epochs': 1,
    'train_batch_size': 128,
    'eval_batch_size': 128,
    'metadata_path': './data/tft/metadata',
    'model_type': 'sepcnn'
}

hooks = [tf_debug.LocalCLIDebugHook()]

shutil.rmtree('./model_trained', ignore_errors=True)

estimator.train(input_fn=read_dataset(args, mode=tf.estimator.ModeKeys.TRAIN), steps=1000, hooks=hooks)