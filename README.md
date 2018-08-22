# Tensorflow Boilerplate

Provides boilerplate for a production Tensorflow project, including serving using the `tf.estimatator` API, hyperparameter tuning etc. Can train and serve both on-premise and on the cloud.

**Note: this code will not work on Windows because Apache Beam can only be installed on Python 2 at the moment, but Tensorflow cannot be installed on Python 2 on Windows.**

## Workflow

With this sample project, the primary interface will be the Jupyter Notebook. This allows a user to interactively develop the model (although when using the Tensorflow APIs such as `tf.estimator` and `tf.transform`, having an interactive kernel may not be that useful).

This is split into 3 notebooks:

1. The first notebook is for data exploration/profiling and pre-processing. The pre-processing in this notebook should be minimal - pre-processing should be done in the second notebook. There, by using `tf.transform`, a pipeline to transform the data as well as a transform function that can be used during serving can be created. This notebook will split the data.
1. The second notebook implements the pre-processing pipeline in `tf.transform`, filtering away invalid data and performing any pre-processing required.
1. The third notebook implements the model itself, and packages up the model into a package.
1. The fourth notebook processes eval data for Tenforflow Model Analysis. 
1. Finally, the fifth notebook uses Tensorflow Model Analysis to analyse the performance of the trained model, slicing metrics by specific columns.

## Key APIs Used

- `pandas` is great for reading in data to Python, providing readers for many common formats of files like csv and Excel.
- `pandas-profiling` is great for exploratory data analysis, providing descriptive statistics for all columns in the dataset, which is crucial in finding out how the data looks like, if there are any outliers, and if there are any invalid or unusual data.
- `tf.transform` is harder to use than just transforming data using pandas, but is more scalable, and helps to avoid having to engineer custom serialisation and deserialisation of scalars/objects that have to be calculated on the training data and applied on eval / test / prediction data. For example, it helps to store the mean and standard deviation of a numeric column to scale it later, and also store tokenizers for text data.
- `tf.estimator` provides a way to define models that can be trained in a distributed fashion, without having to worry about how to distribute the training. It is harder than some other APIs like `tf.keras`, but integrates better with the other APIs like `tf.transform` and `Tensorflow Serving`.
- `Tensorflow Model Analysis` helps to drill down into the model performance, slicing metrics across different sub-categories of the dataset, surfacing problems such as bias.
- `Tensorflow Serving` provides an interface to serve predictions after the model is trained, that scales up to Google's needs. An alternative would be to use `Flask`, which has the added benefits of being easier to configure the API spec. However, it is less scalable. With proper use of `tf.transform` and `Tensorflow Serving`, the inputs to the API immediately go into a computational graph, avoiding the overhead of transferring data between Python and C for pre-processing.

## Pre-requisites

To run this code, you would need Python 2 (for Apache Beam). You would need the following Python packages:

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

- Add slicing to Tensorflow Model Analysis
- Support keras models
- Add hyperparameter tuning
- Use [tfma's example of packaging the model](https://github.com/tensorflow/model-analysis/tree/master/examples/chicago_taxi)
