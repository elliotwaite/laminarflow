"""Classes for writing to and reading from TFRecord datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import multiprocessing
import os

import numpy as np
import tensorflow as tf


def tfrecord_path_to_metadata_path(tfrecord_path):
  """Generate the path where the metadata JSON file will be saved.

  Args:
    tfrecord_path: String. The path where the TFRecord file will be saved.

  Returns:
    String. The path where the metadata JSON file will be saved.

  Raises:
    ValueError: If the specified `tfrecord_path` ends with '.json'.
  """
  root, ext = os.path.splitext(tfrecord_path)
  if ext == '.json':
    raise ValueError('The TFRecord path must not end with ".json". The'
                     'TFRecord path you specified was: %s' % tfrecord_path)
  return root + '.json'


class DatasetWriter(object):
  """A class for writing Examples or SequenceExamples to a TFRecord file.

  When you save an Example to the TFRecord file, the type and shape of the
  Example's Features are writen to a separate metadata file. That type and
  shape data is used later by the SmartTFRecordDataset class when reading
  from the TFRecord file.
  """
  def __init__(self, tfrecord_path):
    """Initialize the writer.

    If a file already exists at the specified `tfrecord_path` or the
    generated `metadata_path`, they will be deleted.

    Args:
      tfrecord_path: String. The path where the TFRecord file will be saved.
    """
    self.metadata_path = tfrecord_path_to_metadata_path(tfrecord_path)
    if os.path.exists(tfrecord_path):
      os.remove(tfrecord_path)
    if os.path.exists(self.metadata_path):
      os.remove(self.metadata_path)

    # Create the TFRecord file's parent directory if it does not exits.
    tfrecord_dir = os.path.dirname(tfrecord_path)
    if not os.path.exists(tfrecord_dir):
      os.makedirs(tfrecord_dir)

    self.writer = tf.python_io.TFRecordWriter(tfrecord_path)
    self.metadata = None

  @staticmethod
  def _get_type(val):
    """Get the type of a value."""
    dtype = np.array(val).dtype
    if dtype == np.uint8:
      return 'bytes'
    if np.issubdtype(dtype, np.floating):
      return 'float'
    if np.issubdtype(dtype, np.integer):
      return 'int64'
    raise ValueError('Invalid feature type: %s' % dtype.name)

  @staticmethod
  def _get_shape(val, is_sequence_feature=False):
    """Get the shape of a value."""
    shape = list(np.array(val).shape)
    if is_sequence_feature:
      if len(shape) < 1:
        raise(ValueError('SequenceExample feature values must have a rank of '
                         'at least 1. The provided value had a rank of %d.'
                         % len(shape)))
      shape[0] = -1
    elif len(shape) == 1 and shape[0] == 1:
      shape = []
    return shape

  def _build_and_write_metadata(self, features, context_features,
                                sequence_features):
    """Build the self.metadata dict, and write it to a JSON file."""
    if features:
      self.metadata = {
        'type': 'Example',
        'features': {
          key: {
            'type': self._get_type(val),
            'shape': self._get_shape(val)
          }
          for key, val in features.items()
        }
      }
    else:
      self.metadata = {
        'type': 'SequenceExample'
      }
      if context_features:
        self.metadata['context_features'] = {
          key: {
            'type': self._get_type(val),
            'shape': self._get_shape(val)
          }
          for key, val in context_features.items()
        }
      if sequence_features:
        self.metadata['sequence_features'] = {
          key: {
            'type': self._get_type(val),
            'shape': self._get_shape(val, is_sequence_feature=True)
          }
          for key, val in sequence_features.items()
        }

    # Write the metadata to a JSON file in an easily readable format.
    s = '{\n    "type": "' + self.metadata['type'] + '",'
    for features_type in ['features', 'context_features', 'sequence_features']:
      if features_type in self.metadata:
        s += '\n    "' + features_type + '": {'
        for key, val in sorted(self.metadata[features_type].items()):
          s += '\n        "' + key + '": ' + json.dumps(val) + ','
        s = s[:-1]  # Remove the comma after the last item.
        s += '\n    },'
    s = s[:-1]  # Remove the comma after the last item.
    s += '\n}'
    with open(self.metadata_path, 'w') as f:
      f.write(s)

  @staticmethod
  def _feature_to_example_feature(val, type_):
    """Return an Example Feature given the value and type."""
    val = np.array(val).flatten()
    if type_ == 'bytes':
      val = val.tobytes()
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=val))
    if type_ == 'float':
      return tf.train.Feature(float_list=tf.train.FloatList(value=val))
    if type_ == 'int64':
      return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
    raise ValueError('Invalid type: %s' % type_)

  @staticmethod
  def _feature_to_example_feature_list(val, type_):
    """Return an Example FeatureList given the value and type."""
    val = np.array(val)
    val = val.reshape(val.shape[0], -1)
    if type_ == 'bytes':
      val = val.tobytes()
      feature = [tf.train.Feature(bytes_list=tf.train.BytesList(value=row))
                 for row in val]
    elif type_ == 'float':
      feature = [tf.train.Feature(float_list=tf.train.FloatList(value=row))
                 for row in val]
    elif type_ == 'int64':
      feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=row))
                 for row in val]
    else:
      raise ValueError('Invalid type: %s' % type_)

    return tf.train.FeatureList(feature=feature)

  def write(self, features=None, context_features=None, sequence_features=None):
    """Write an Example or SequenceExample to the TFRecord file.

    If `features` are passed in, an Example will be created. Otherwise, if
    `context_features`, or `sequence_features`, or both are passed in,
    a SequenceExample will be created. At least one of `features`,
    `context_features`, or `sequence_features` must be passed in.

    Args:
      features: A dict of key value pairs where the key is the feature's
          name and the value is the feature's value (int, float, list,
          np.array). The shape of the value can be multidimensional,
          but must be the same between Examples.
      context_features: A dict of key value pairs where the key is the context
          feature's name and the value is the context feature's value (int,
          float, list, np.array). The shape of the value can be
          multidimensional, but must be the same between SequenceExamples.
      sequence_features: A dict of key value pairs where the key is the
          sequence feature's name and the value is the sequence feature's
          value (int, float, list, np.array). The shape of the value can be
          multidimensional, and must have a rank of at least 1. The length of
          the first dimension can be variable, but the rest of the shape must
          be the same between SequenceExamples.

    Raises:
      ValueError: If `features`,`context_features`, and `sequence_features` are
      all None.
    """
    if not self.metadata:
      self._build_and_write_metadata(features, context_features,
                                     sequence_features)

    if not features and not context_features and not sequence_features:
      raise ValueError('`features`, `context_features`, and '
                       '`sequence_features` cannot all be None.')

    if features:
      # Create an Example.
      feature_map = {
        key: self._feature_to_example_feature(
            val, self.metadata['features'][key]['type'])
        for key, val in features.items()
      }
      example = tf.train.Example(
          features=tf.train.Features(feature=feature_map))

    else:
      # Create a SequenceExample.
      if context_features:
        feature_map = {
          key: self._feature_to_example_feature(
              val, self.metadata['context_features'][key]['type'])
          for key, val in context_features.items()
        }
        context = tf.train.Features(feature=feature_map)
      else:
        context = None

      if sequence_features:
        feature_list_map = {
          key: self._feature_to_example_feature_list(
              val, self.metadata['sequence_features'][key]['type'])
          for key, val in sequence_features.items()
        }
        feature_lists = tf.train.FeatureLists(feature_list=feature_list_map)
      else:
        feature_lists = None

      example = tf.train.SequenceExample(
          context=context,
          feature_lists=feature_lists)

    self.writer.write(example.SerializeToString())

  def close(self):
    """Close the writer."""
    self.writer.close()

  def __enter__(self):
      return self

  def __exit__(self, *exc):
      self.close()


class DatasetReader(object):
  """A class for piping Examples from a TFRecord file into an Estimator.

  When calling an Estimators train, eval, or predict methods, pass in this
  class's input_fn method as the input_fn parameter and it will read Examples
  from the TFRecord file using the parameters specified when initializing an
  instance of this class.
  """

  def __init__(self, tfrecord_path, batch_size=1, num_parallel_batches=None,
               shuffle_buffer_size=None, repeat=None, prefetch_buffer_size=1):
    """Initialize the database object.

    Store the initialization parameters and read in the metadata from the
    metadata file."""
    if not tfrecord_path.endswith('.tfrecord'):
      raise ValueError('The TFRecord path must end with ".tfrecord", however '
                       'the path you specified was: %s' % tfrecord_path)
    self.tfrecord_path = tfrecord_path
    self.batch_size = batch_size
    self.num_parallel_batches = (num_parallel_batches if num_parallel_batches
                                 else multiprocessing.cpu_count())
    self.shuffle_buffer_size = shuffle_buffer_size
    self.repeat = repeat
    self.prefetch_buffer_size = prefetch_buffer_size

    self.metadata_path = tfrecord_path_to_metadata_path(tfrecord_path)
    with open(self.metadata_path) as f:
      self.metadata = json.load(f)

    if self.metadata['type'] == 'Example':
      self.features_parser_config = {
        key: self._get_feature_parser_config(val)
        for key, val in self.metadata['features'].items()
      }

    if self.metadata['type'] == 'SequenceExample':
      if 'context_features' in self.metadata:
        self.context_features_parser_config = {
          key: self._get_feature_parser_config(val)
          for key, val in self.metadata['context_features'].items()
        }
      else:
        self.context_features_parser_config = None

      if 'sequence_features' in self.metadata:
        self.sequence_features_parser_config = {
          key: self._get_feature_parser_config(val, is_sequence_feature=True)
          for key, val in self.metadata['sequence_features'].items()
        }
      else:
        self.sequence_features_parser_config = None

  @staticmethod
  def _get_feature_parser_config(feature_metadata, is_sequence_feature=False):
    """Get the parsing configuration for the feature, given its metadata.

    Returns: A FixedLenFeature if the shape is constant, or a
        FixedLenSequenceFeature if the first dimension of the shape is of
        variable length.
    """
    if feature_metadata['type'] == 'bytes':
      tf_type = tf.string
    elif feature_metadata['type'] == 'float':
      tf_type = tf.float32
    elif feature_metadata['type'] == 'int64':
      tf_type = tf.int64
    else:
      raise ValueError('Invalid metadata type: %s' % feature_metadata['type'])

    if is_sequence_feature:
      return tf.FixedLenSequenceFeature(feature_metadata['shape'][1:], tf_type)
    else:
      return tf.FixedLenFeature(feature_metadata['shape'], tf_type)

  # def _decode_if_needed(self, key, val):
  #   """Reshape the tensor to the metadata's shape if it's multidimensional."""
  #   shape = self.metadata[key]['shape']
  #   if len(shape) > 1:
  #     val = tf.reshape(val, shape)
  #   return val

  @staticmethod
  def _decode_and_reshape_if_needed(val, shape):
    """Decode the value to unit8 if it is a byte string, and reshape the
    tensor if it's shape multidimensional."""
    if val.dtype == tf.string:
      val = tf.decode_raw(val, tf.uint8)
    if len(shape) > 1:
      val = tf.reshape(val, shape)
    return val

  def _parser(self, serialized_example):
    """Deserialize the Example and return a features dict and a label."""
    if self.metadata['type'] == 'Example':
      # Parse an Example.
      features = tf.parse_single_example(serialized_example,
                                         self.features_parser_config)
      for key in features:
        features[key] = self._decode_and_reshape_if_needed(
            features[key], self.metadata['features'][key]['shape'])

    else:
      # Parse a SequenceExample.
      features, sequence_features = tf.parse_single_sequence_example(
          serialized_example, self.context_features_parser_config,
          self.sequence_features_parser_config)

      for key in features:
        features[key] = self._decode_and_reshape_if_needed(
            features[key], self.metadata['context_features'][key]['shape'])
      for key in sequence_features:
        features[key] = self._decode_and_reshape_if_needed(
            sequence_features[key],
            self.metadata['sequence_features'][key]['shape'])

      features.update(sequence_features)

      features['length'] = tf.shape(
        sequence_features[sequence_features.keys()[0]])[0]

    label = features.pop('label', None)
    if label is None:
      label = features.pop('labels', None)

    return features, label

  def _get_padded_shapes(self):
    features_padded_shapes = {}
    if 'context_features' in self.metadata:
      features_padded_shapes.update({
        key: val['shape']
        for key, val in self.metadata['context_features'].items()
      })
    if 'sequence_features' in self.metadata:
      features_padded_shapes.update({
        key: val['shape']
        for key, val in self.metadata['sequence_features'].items()
      })
    features_padded_shapes['length'] = []
    label_padded_shape = (features_padded_shapes.pop('label', None) or
                          features_padded_shapes.pop('labels', None))

    return features_padded_shapes, label_padded_shape

  def input_fn(self):
    """The input_fn to be passed to an Estimator.

    This method will add nodes and ops to the graph that will parse Examples
    from the TFRecord file, batch them together, shuffle them if specified,
    and repeat the dataset multiple epochs if specified.

    Returns:
      features: A dict of key value pairs where the key is the feature's name
          and the value is a Tensor of a batch of that feature.
      labels: A Tensor of a batch of labels.
    """
    dataset = tf.data.TFRecordDataset(self.tfrecord_path)

    if self.metadata['type'] == 'Example':
      # Get a batch of Examples.
      dataset = dataset.apply(tf.contrib.data.map_and_batch(
          self._parser, self.batch_size, self.num_parallel_batches))
    else:
      # Get a dynamically padded batch of SequenceExamples
      dataset = dataset.map(self._parser)
      dataset = dataset.padded_batch(self.batch_size,
                                     padded_shapes=self._get_padded_shapes())

    if self.shuffle_buffer_size and self.repeat != 1:
      dataset.apply(tf.contrib.data.shuffle_and_repeat(
          self.shuffle_buffer_size, self.repeat))
    else:
      if self.shuffle_buffer_size:
        dataset = dataset.shuffle(self.shuffle_buffer_size)
      if self.repeat != 1:
        dataset = dataset.repeat(self.repeat)

    dataset = dataset.prefetch(self.prefetch_buffer_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
