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