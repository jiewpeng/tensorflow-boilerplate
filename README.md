# Tensorflow Boilerplate

Provides boilerplate for a production Tensorflow project, including serving using the `tf.estimatator` API, hyperparameter tuning etc. Can train and serve both on-premise and on the cloud.

**Note: this code will not work on Windows because Apache Beam can only be installed on Python 2 at the moment, but Tensorflow cannot be installed on Python 2 on Windows.**

## Workflow

With this sample project, we will be working primarily from a Jupyter Notebook. This allows us to interactively develop the model (although when using the Tensorflow APIs such as `tf.estimator` and `tf.transform`, having an interactive kernel may not be that useful).

This is split into 3 notebooks:

1. The first notebook is for data exploration/profiling and pre-processing. The pre-processing in this notebook should be minimal - pre-processing should be done in the second notebook. There, by using `tf.transform`, we can create a pipeline to transform the data as well as a transform function that can be used during serving. This notebook will split the data.
1. The second notebook implements the pre-processing pipeline in `tf.transform`, filtering away invalid data and performing any pre-processing required.
1. Finally, the third notebook implements the model itself, and packages up the model into a package.

## Pre-requisites

To run this code, we would need Python 2 (for Apache Beam). We would need the following Python packages:

- tensorflow
- tensorflow-serving
- tensorflow-transform
- tensorflow-hub
- snappy
- apache-beam
- pandas
- pandas-profiling

## Dataset Information

The sample data in this repo is the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), which has been collected from various sources.

## TODO

- Use Tensorflow Model Analysis to produce dashboard of model performance analysis 
- Add hyperparameter tuning
- Add documentation for key APIs used and reasons for using them
- Use [tfma's example of packaging the model](https://github.com/tensorflow/model-analysis/tree/master/examples/chicago_taxi)
