"""Train a model to predict a sequence XOR output values.

In this example we use LaminarFlow's DatasetWriter to write TensorFlow
SequenceExamples to a TFRecord file. Then we use LaminarFlow's DatasetReader
to create an input pipeline for that dataset, and use it to train a model.

First we create a training dataset and test dataset of sequences of XOR data.
Then we train a model using those datasets.
Then we make predictions using the trained model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import shutil

import laminarflow as lf
import numpy as np
import tensorflow as tf

TRAIN_TFRECORD_PATH = '/tmp/xor/data/train.tfrecord'
TEST_TFRECORD_PATH = '/tmp/xor/data/test.tfrecord'
DATASET_SIZE = 1000
SEQUENCE_MAX_STEPS = 20
MODEL_DIR = '/tmp/xor/model'
PARAMS = {
  'batch_size': 128,
  'lstm_size': 2,
  'num_classes': 2,
  'learning_rate': 0.1,
}
RESTART_TRAINING = True
NUM_TRAINING_STEPS = 1000


def create_datasets():
  """Create the training and testing datasets.

  We'll use 1000 examples of sequences of XOR data as our dummy data and
  store 90% of it in our training dataset and 10% of it in our test dataset.
  Sequences of XOR data are random sequences of 1s and 0s, where the label for
  that step is the XOR output of the last two values. And the XOR output for
  the first step will be as if the value before it was 0. For example:
  Sequence: 1 0 0 1 0 1 1 1 0 0 0 0
  Labels:   1 1 0 1 1 1 0 0 1 0 0 0
  The length of the sequences will be between 1 and 20 steps long.
  """
  # Create our dataset.
  xor_dataset = []
  for _ in range(DATASET_SIZE):
    inputs = []
    labels = []
    last_input = 0
    sequence_length = random.randint(1, SEQUENCE_MAX_STEPS)
    for _ in range(sequence_length):
      new_input = random.randint(0, 1)
      inputs.append(float(new_input))  # Our inputs will be floats.
      labels.append(new_input ^ last_input)  # And our labels will be ints.
      last_input = new_input
    xor_dataset.append((inputs, labels))

  # Write our dataset to a TFRecord file.
  train_writer = lf.DatasetWriter(TRAIN_TFRECORD_PATH)
  test_writer = lf.DatasetWriter(TEST_TFRECORD_PATH)

  num_training_examples = int(len(xor_dataset) * 0.9)

  for i, (inputs, labels) in enumerate(xor_dataset):
    writer = train_writer if i < num_training_examples else test_writer
    writer.write(sequence_features={
      'inputs': inputs,
      'label': labels
    })

  train_writer.close()
  test_writer.close()


def model_fn(features, labels, mode, params):
  """Our model function, a simple single hidden layer feedforward network."""
  max_sequence_length = tf.shape(features['inputs'])[1]

  # The RNN expects a shape of [batch_size, sequence_length, input_size],
  # but features['inputs'] has a shape of [batch_size, sequence_length],
  # so we expand it to [batch_size, sequence_length, 1].
  expanded_inputs = tf.expand_dims(features['inputs'], axis=2)

  # We'll use a basic LSTM.
  cell = tf.nn.rnn_cell.BasicLSTMCell(params['lstm_size'])

  outputs, state = tf.nn.dynamic_rnn(cell, expanded_inputs,
                                     dtype=tf.float32, swap_memory=True)

  logits = tf.layers.dense(outputs, params['num_classes'])
  predictions = tf.argmax(logits, axis=2)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # For predictions, we return the most likely class index for each step.
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Create our weights mask from our sequence lengths. `features['lengths']`
  # is the length of each sequence in the batch before it was padded.
  # The lengths feature is automatically generated during the padding and
  # batching step of LaminarFlow's DatasetReader input pipeline.
  range_ = tf.range(max_sequence_length)
  range_row = tf.expand_dims(range_, axis=0)
  lengths_col = tf.expand_dims(features['lengths'], axis=1)
  mask = tf.less(range_row, lengths_col)
  weights = tf.cast(mask, dtype=tf.float32)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits,
                                                weights=weights)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Log the accuracy of the model on the test dataset.
    eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(labels=labels,
                                      predictions=predictions,
                                      weights=weights)}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

  optimizer = tf.train.AdamOptimizer(params['learning_rate'])
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  # Also log the accuracy of the model on the training dataset.
  accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions,
                                                 labels)))
  tf.summary.scalar('accuracy', accuracy)

  return tf.estimator.EstimatorSpec(mode=mode,
                                    loss=loss,
                                    train_op=train_op)


def train():
  """Train the model.

  You can view the training progress with TensorBoard by running:
    tensorboard --logdir=/tmp/xor/model
  """
  tf.logging.set_verbosity(tf.logging.INFO)

  # To restart training, delete any previously created checkpoints.
  if RESTART_TRAINING and os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

  # Create our dataset input pipelines for our training and test datasets.
  train_dataset = lf.DatasetReader(TRAIN_TFRECORD_PATH,
                                   batch_size=PARAMS['batch_size'],
                                   shuffle_buffer_size=None)

  test_dataset = lf.DatasetReader(TEST_TFRECORD_PATH,
                                  batch_size=PARAMS['batch_size'],
                                  repeat=1)

  # Create an estimator for our model.
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=MODEL_DIR,
                                     params=PARAMS)

  # Specify our dataset input pipelines in our estimator specs.
  train_spec = tf.estimator.TrainSpec(input_fn=train_dataset.input_fn,
                                      max_steps=NUM_TRAINING_STEPS)

  eval_spec = tf.estimator.EvalSpec(input_fn=test_dataset.input_fn)

  # Train and evaluate our model.
  tf.estimator.train_and_evaluate(estimator=estimator,
                                  train_spec=train_spec,
                                  eval_spec=eval_spec)


def predict():
  """Make predictions with the model."""
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=MODEL_DIR,
                                     params=PARAMS)

  # Well make predictions for 4 random 10-step input sequences.
  inputs = np.array(
      [[0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
       [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
  expectations = np.array(
      [[0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
       [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

  input_fn = tf.estimator.inputs.numpy_input_fn(
      {'inputs': inputs.astype(np.float32)}, shuffle=False)
  predictions = estimator.predict(input_fn=input_fn)
  zipped = zip(inputs, expectations, predictions)
  for (input_, expectation, prediction) in zipped:
    print('      Inputs: %s\n'
          'Expectations: %s\n'
          ' Predictions: %s\n' % (input_, expectation, prediction))


if __name__ == '__main__':
    create_datasets()
    train()
    predict()
