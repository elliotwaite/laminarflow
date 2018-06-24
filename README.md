# LaminarFlow

Streamline your TensorFlow workflow.

## TFRecord Datasets

LaminarFlow has two classes for writing to and reading from TFRecord datasets, `DatasetWriter` and `DatasetReader`.

When creating your datasets with `DatasetWriter`, you can pass in raw Python or Numpy data, and it will automatically get converted into tf.Examples or tf.SequenceExamples and be written to the TFRecord file.

Then when reading from the TFRecord file, `DatasetReader` takes care of creating the input pipeline that will parse your stored tf.Examples or tf.SequenceExamples, prepare them as needed (batching, padding, shuffling, etc.), then pass them to your TensorFlow Estimator. 

The code looks like this. In this example, we'll train a model to predict an XOR circuit's output value. First, create your TFRecord datasets.

```python
import laminarflow as lf

train_writer = lf.DatasetWriter('/tmp/train.tfrecord')
test_writer = lf.DatasetWriter('/tmp/test.tfrecord')

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
Simply create a `DatasetWriter`, then call the `write` method on it, passing in a dictionary where the keys are the feature names and the values are the feature values. The values can be Numpy arrays or any values that can be converted into Numpy arrays, such as Python ints, floats, or lists of ints or floats.
 
When you are done writing data with a Writer, call the `close()` method on it.

The data will be written to a TFRecord file and the shapes and data types of your features will be stored in a separate metadata file (which will have the same filename as the TFRecord file, except the extension will be changed to '.json').

```
tmp/
├── test.json
├── test.tfrecord
├── train.json
└── train.tfrecord
```

Then you can train a model on your datasets.

```python
train_dataset = lf.DatasetReader('/tmp/train.tfrecord')
test_dataset = lf.DatasetReader('/tmp/test.tfrecord')

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

Calling `lf.DatasetReader('/tmp/train.tfrecord')` creates a dataset using the TFRecord file and its corresponding metadata JSON file. The path to the metadata JSON file `/tmp/train.json` is inferred from the TFRecord path.

The created dataset has an `input_fn` method that you can pass in as the input function to a TensorFlow Estimator. The `input_fn` method automatically creates the input pipeline for your dataset, implementing the recommended best practices as outlined in TensorFlow's [Input Pipline Performance Guide](https://www.tensorflow.org/performance/datasets_performance).

### Additional Features

You can create a `DatasetWriter` using a `with` statement, in which case you wouldn't have to explicitly call the `close()` method.

```python
with lf.DatasetWriter('/tmp/train.tfrecord') as train_writer:
  train_writer.write({
    'input': [1.4, 1.4, 2.1],
    'label': 3
  })
```