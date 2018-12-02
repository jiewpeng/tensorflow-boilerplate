
import tensorflow as tf


LABEL_TO_INDEX = tf.contrib.lookup.index_table_from_file(
    './data/misc/labels.txt',
    default_value=-1,
    num_oov_buckets=1
)

INDEX_TO_LABEL = tf.contrib.lookup.index_to_string_table_from_file(
    './data/misc/labels.txt',
    default_value='ham'
)

N_CLASSES = 2
LABEL_VOCABULARY = ['ham', 'spam']
