import traceback
import argparse
import json
import os

import model

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train_data_paths',
        help = 'GCS or local path to training data',
        required = True
    )
    parser.add_argument(
        '--train_batch_size',
        help = 'Batch size for training steps',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--eval_batch_size',
        help = 'Batch size for evaluation steps',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--num_epochs',
        help = 'Epochs to run the training job for',
        type = int,
        default = 50
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Number of steps to run evalution for at each checkpoint',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--eval_data_paths',
        help = 'GCS or local path to evaluation data',
        required = True
    )
    # TensorFlow Transform args
    parser.add_argument(
        '--metadata_path',
        help = 'GCS or local path to transformed metadata if using TFT',
        default = '../../data/tft/metadata'
    )
    # Training arguments
    parser.add_argument(
        '--model_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )
    # Eval arguments
    parser.add_argument(
        '--eval_delay_secs',
        help = 'How long to wait before running first evaluation',
        default = 10,
        type = int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help = 'Minimum number of training steps between evaluations',
        default = 1,
        type = int
    )
    # Model Specific arguments
    parser.add_argument(
        '--model_type',
        help='Type of ML model, either "linear" or "dnn"',
        default='linear',
        type=str
    )
    parser.add_argument(
        '--embedding_type',
        help='Embedding to use, one of "nnlm", "universal-sentence-encoder", "elmo"',
        default='universal-sentence-encoder',
        type=str
    )
    parser.add_argument(
        '--learning_rate',
        help='Learning rate',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--hidden_units',
        help='Hidden units of the DNN model, separated by space e.g. "128 64"',
        type=str
    )
    parser.add_argument(
        '--dropout',
        help='Dropout rate (between 0 and 1)',
        default=0.0,
        type=float
    )
    parser.add_argument(
        '--l1_regularization_strength',
        help='L1 regularisation strength; controls how sparse the linear model will be',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--l2_regularization_strength',
        help='L2 regularisation strength; controls the magnitude of the weights in the linear model',
        default=0.01,
        type=float
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    output_dir = arguments['model_dir']

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # Run the training job:
    try:
        model.train_and_evaluate(arguments)
    except:
        traceback.print_exc()