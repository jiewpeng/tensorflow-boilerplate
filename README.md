# Tensorflow Boilerplate

Provides boilerplate for a production Tensorflow project, including serving using the `tf.estimatator` API, hyperparameter tuning etc. Can train and serve both on-premise and on the cloud.

**Note: this code will not work on Windows because it requires Python 2, since Apache Beam does not work on Python 3. Tensorflow cannot be installed on Python 2 on Windows.**

## Workflow

With this sample project, we will be working primarily from a Jupyter Notebook. This allows us to interactively develop the model (although when using the Tensorflow APIs such as `tf.estimator` and `tf.transform`, having an interactive kernel may not be that useful).

This is split into 3 notebooks:

1. The first notebook is for data exploration/profiling and pre-processing. The pre-processing in this notebook should be minimal - pre-processing should be done in the second notebook. There, by using `tf.transform`, we can create a pipeline to transform the data as well as a transform function that can be used during serving. This notebook will split the data.
1. The second notebook implements the pre-processing pipeline in `tf.transform`, filtering away invalid data and performing any pre-processing required.
1. Finally, the third notebook implements the model itself, and packages up the model into a package.