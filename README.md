# LaminarFlow

Streamline your TensorFlow workflow.

## Installation

```
pip install laminarflow
```

## Usage

#### TFRecord Datasets

LaminarFlow has two classes for writing to and reading from TFRecord datasets, `DatasetWriter` and `DatasetReader`.

When creating your datasets with `DatasetWriter`, you can pass in raw Python or Numpy data, and it will automatically get converted into TensorFlow Examples or SequenceExamples and be written to a TFRecord file.

Then when reading from the TFRecord file, `DatasetReader` takes care of creating the input pipeline that will parse your stored Examples or SequenceExamples, prepare them as needed (batching, padding, shuffling, etc.), then pass them to your TensorFlow Estimator, implementing the recommended best practices as outlined in TensorFlow's [Input Pipline Performance Guide](https://www.tensorflow.org/performance/datasets_performance).

To demonstrate, we'll create some datasets.

```python
import laminarflow as lf

train_writer = lf.DatasetWriter('data/train.tfrecord')
test_writer = lf.DatasetWriter('data/test.tfrecord')

train_writer.write({
  'input': [3.1, 4.1, 5.9],
  'label': 2
})

train_writer.write({
  'input': [2.7, 1.8, 2.8],
  'label': 1
})

test_writer.write({
  'input': [0.1, 1.2, 3.5],
  'label': 8
})

train_writer.close()
test_writer.close()
```
We create a `DatasetWriter`, then call the `write` method on it for each TensorFlow Example or SequenceExample we want to add to the dataset. When we call the `write` method, we pass in a dictionary where the keys are the feature names and the values are the feature values. The values can be Numpy arrays or any values that can be converted into Numpy arrays, such as Python ints, floats, or lists of ints or floats. The shape of the values can be multidimensional, but must be the same between Examples. Creating SequenceExamples, which support variable length data, is discussed below.
 
When we are done writing data with a Writer, we call the `close()` method on it.

The data will be written to a TFRecord file and the shapes and data types of your features will be stored in a separate metadata JSON file, which will have the same filename as the TFRecord file, except the extension will be changed to '.json'.

```
data/
├── test.json
├── test.tfrecord
├── train.json
└── train.tfrecord
```

We can then train a model on our datasets.

```python
train_dataset = lf.DatasetReader('data/train.tfrecord')
test_dataset = lf.DatasetReader('data/test.tfrecord')

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  model_dir=MODEL_DIR,
  params=PARAMS)

train_spec = tf.estimator.TrainSpec(
    input_fn=train_dataset.input_fn,
    max_steps=1000)
    
eval_spec = tf.estimator.EvalSpec(
    input_fn=test_dataset.input_fn)
    
tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=train_spec,
    eval_spec=eval_spec)
```

Calling `lf.DatasetReader('data/train.tfrecord')` creates a dataset using the TFRecord file and its corresponding metadata JSON file. The path to the metadata JSON file `data/train.json` is inferred from the TFRecord path.

The created dataset has an `input_fn` method that you can pass in as the input function to a TensorFlow Estimator. The `input_fn` method automatically creates the input pipeline for your dataset.

For a more complete example of creating datasets, training a model, and making predictions with that model, check out: [xor.py](examples/xor.py)

#### Using a `with` Statement

A `DatasetWriter` can also be created using a `with` statement, in which case the `close()` method does not need to be called.

```python
with lf.DatasetWriter('data/train.tfrecord') as train_writer:
  train_writer.write({
    'input': [1.4, 1.4, 2.1],
    'label': 3
  })
```

#### SequenceExamples

The default behavior of the `write` method is to write a TensorFlow Example. To write a SequenceExample, instead of passing in features to the first parameter of the `write` method, pass in features using the `context_features` and `sequence_features` parameters.

```python
train_writer.write(
  context_features={
    'category': 7
  },
  sequence_features={
    'inputs': [[1.4, 0.0], [1.4, 0.0], [1.4, 0.0]],
    'labels': [3, 5, 3]
  })
  
train_writer.write(
  context_features={
    'category': 5
  },
  sequence_features={
    'inputs': [[1.4, 0.0], [1.4, 0.0]],
    'labels': [3, 5]
  })
```

Passing in `context_features` is optional, but if used, their values must have the same shape between SequenceExamples, similar to Example features.

The shape of the `sequence_features` values must have a rank of at least 1. The length of the first dimension must be the same for all `sequence_features` within a SequenceExample, but can vary between SequenceExamples. And the lengths of the rest of the dimensions can vary between features, but must be the same between SequenceExamples.

When a batch of SequenceExamples is created, any sequences that are shorter than the longest sequence will be padded with zeros.

The length of each sequence will be extracted from the data as one of the steps in the input pipeline when reading from the dataset. The lengths of the sequences will be made available as one of the feature values passed into the model_fn, `features['lengths']`. It will be a batch size length list of ints, that are the lengths of each of the sequences in the batch before that sequence was possibly padded with zeros.