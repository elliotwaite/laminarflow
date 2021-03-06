"""Train a model to predict XOR output values.

In this example we use LaminarFlow's DatasetWriter to write TensorFlow Examples
to a TFRecord file. Then we use LaminarFlow's DatasetReader to create an
input pipeline for that dataset, and then use it to train a model.

First we create a training dataset and test dataset of XOR data.
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
MODEL_DIR = '/tmp/xor/model'
PARAMS = {
  'batch_size': 128,
  'hidden_units': 2,
  'num_classes': 2,
  'learning_rate': 0.1,
}
RESTART_TRAINING = True
NUM_TRAINING_STEPS = 1000


def create_datasets():
  """Create the training and testing datasets.

  We'll use 1000 examples of XOR data as our dummy data and store 90% of it
  in our training dataset and 10% of it in our test dataset.
  """
  # Create our dataset.
  xor_dataset = []
  for _ in range(DATASET_SIZE):
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    inputs = (float(a), float(b))  # Our inputs will be floats.
    label = a ^ b  # And our labels will be ints.
    xor_dataset.append((inputs, label))

  # Write our dataset to a TFRecord file.
  train_writer = lf.DatasetWriter(TRAIN_TFRECORD_PATH)
  test_writer = lf.DatasetWriter(TEST_TFRECORD_PATH)

  num_training_examples = int(len(xor_dataset) * 0.9)

  for i, (inputs, label) in enumerate(xor_dataset):
    writer = train_writer if i < num_training_examples else test_writer
    writer.write({
      'inputs': inputs,
      'label': label
    })

  train_writer.close()
  test_writer.close()


def model_fn(features, labels, mode, params):
  """Our model function, a simple single hidden layer feedforward network."""
  hidden = tf.layers.dense(inputs=features['inputs'],
                           units=params['hidden_units'],
                           activation=tf.nn.elu)

  # We could also make the output a single unit, and use the
  # tf.losses.sigmoid_cross_entropy() loss function, but for this demo we'll
  # use a more general n-class classification model structure and use the
  # tf.losses.sparse_softmax_cross_entropy() loss function.
  logits = tf.layers.dense(hidden, params['num_classes'])

  predictions = tf.argmax(logits, axis=1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # For predictions, just return the most likely class index.
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Log the accuracy of the model on the test dataset.
    eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

  optimizer = tf.train.AdamOptimizer(params['learning_rate'])
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  # Also log the accuracy of the model on the training dataset.
  accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
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

  # Well make predictions for all 4 possible XOR input combinations.
  inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  expectations = np.array([0, 1, 1, 0])

  input_fn = tf.estimator.inputs.numpy_input_fn(
      {'inputs': inputs.astype(np.float32)}, shuffle=False)
  predictions = estimator.predict(input_fn=input_fn)
  zipped = zip(inputs, expectations, predictions)
  for (input_, expectation, prediction) in zipped:
    print('Input: %s, Expectation: %s, Prediction: %s'
          % (input_, expectation, prediction))


if __name__ == '__main__':
    create_datasets()
    train()
    predict()
