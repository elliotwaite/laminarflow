"""An example that uses LaminarFlow's DatasetWriter and DatasetReader.

First we create a training set and test set of XOR data.
Then we train a model on those datasets.
Then we make predictions with the trained model.
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
MODEL_DIR = '/tmp/xor/model'
PARAMS = {
  'batch_size': 128,
  'hidden_units': 2,
  'num_classes': 2,
  'learning_rate': 0.01,
}
RESTART_TRAINING = True
NUM_TRAINING_STEPS = 5000


def create_datasets():
  """Create the training and testing datasets.

  We'll use 1000 examples of XOR data as our dummy data and store 90% of it
  in our training dataset and 10% of it in our test dataset.
  """
  xor_data = [
    ((0.0, 0.0), 0),
    ((0.0, 1.0), 1),
    ((1.0, 0.0), 1),
    ((1.0, 1.0), 0)
  ]
  xor_dataset = (xor_data * 250)
  random.shuffle(xor_dataset)

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

  logits = tf.layers.dense(hidden, params['num_classes'])

  predictions = tf.argmax(logits, axis=1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Track accuracy when running an evaluation.
    eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

  optimizer = tf.train.AdamOptimizer(params['learning_rate'])
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  # In addition to tracking accuracy during evaluation, also track accuracy
  # during training.
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

  if RESTART_TRAINING and os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

  train_dataset = lf.DatasetReader(TRAIN_TFRECORD_PATH,
                                   batch_size=PARAMS['batch_size'],
                                   shuffle_buffer_size=None)

  test_dataset = lf.DatasetReader(TEST_TFRECORD_PATH,
                                  batch_size=PARAMS['batch_size'],
                                  repeat=1)

  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=MODEL_DIR,
                                     params=PARAMS)

  train_spec = tf.estimator.TrainSpec(input_fn=train_dataset.input_fn,
                                      max_steps=NUM_TRAINING_STEPS)

  eval_spec = tf.estimator.EvalSpec(input_fn=test_dataset.input_fn)

  tf.estimator.train_and_evaluate(estimator=estimator,
                                  train_spec=train_spec,
                                  eval_spec=eval_spec)


def predict():
  """Make predictions with the model."""
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=MODEL_DIR,
                                     params=PARAMS)

  inputs = np.array([[0.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 0.0],
                     [1.0, 1.0]], dtype=np.float32)

  input_fn = tf.estimator.inputs.numpy_input_fn({'inputs': inputs},
                                                shuffle=False)
  predictions = estimator.predict(input_fn=input_fn)
  for (input_, prediction) in zip(inputs, predictions):
    print('Input: %s, Prediction: %s' % (input_, prediction))


if __name__ == '__main__':
    create_datasets()
    train()
    predict()
