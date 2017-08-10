from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops.helper import TrainingHelper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from tensorflow.python.ops.variables import Variable

class WordDropoutTrainingHelper(TrainingHelper):
  """A training helper that adds word dropout."""

  def __init__(self, inputs, sequence_length, embedding, dropout_keep_prob,
               drop_token_id, time_major=False, seed=None,
               dropout_seed=None, name=None):
    with ops.name_scope(name, "WordDropoutTrainingHelper",
                        [embedding, dropout_keep_prob]):
      if callable(embedding):
        self._embedding_fn = embedding
      else:
        self._embedding_fn = (
            lambda ids: embedding_ops.embedding_lookup(embedding, ids))
      self._dropout_keep_prob = ops.convert_to_tensor(
          dropout_keep_prob, name="dropout_keep_prob")
      if self._dropout_keep_prob.get_shape().ndims not in (0, 1):
        raise ValueError(
            "dropout_keep_prob must be either a scalar or a vector. "
            "saw shape: %s" % (self._dropout_keep_prob.get_shape()))
      self._seed = seed
      self._dropout_seed = dropout_seed
      self._drop_token_id = drop_token_id

      super(WordDropoutTrainingHelper, self).__init__(
          inputs=inputs,
          sequence_length=sequence_length,
          time_major=time_major,
          name=name)

  def initialize(self, name=None):
    return super(WordDropoutTrainingHelper, self).initialize(name=name)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with ops.name_scope(name, "WordDropoutTrainingHelper", [time, outputs]):
      sample_ids = math_ops.cast(
          math_ops.argmax(outputs, axis=-1), dtypes.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with ops.name_scope(name, "WordDropoutTrainingHelper",
                        [time, outputs, state, sample_ids]):
      (finished, base_next_inputs, state) = (
          super(WordDropoutTrainingHelper, self).next_inputs(time=time,
                                                             outputs=outputs,
                                                             state=state,
                                                             sample_ids=sample_ids,
                                                             name=name))
      # sample for greedy embedding

      def maybe_dropout():
        """Perform word dropout."""
        keep_noise = random_ops.random_uniform(
                        [self.batch_size-1], seed=self._dropout_seed)
        # always keep the first input token <eos>
        keep_sample_noise = array_ops.concat([[1],keep_noise], axis=0)

        # get masks for keeping and dropout
        keep_sample = (self._dropout_keep_prob > keep_sample_noise) # true/false
        drop_sample = (self._dropout_keep_prob <= keep_sample_noise)

        # get indices (in batch) for keeping and dropout
        where_keeping = math_ops.cast(array_ops.where(keep_sample), dtypes.int32)
        where_dropout = math_ops.cast(array_ops.where(drop_sample), dtypes.int32)
        where_keeping_flat = array_ops.reshape(where_keeping, [-1]) # indices
        where_dropout_flat = array_ops.reshape(where_dropout, [-1])

        # <drop> token tiles for gathering
        drop_ids = array_ops.tile([self._drop_token_id], [self.batch_size])

        # gather inputs in corresponding indices
        keeping_inputs = array_ops.gather(base_next_inputs, where_keeping_flat)
        dropout_inputs_ids = array_ops.gather(drop_ids, where_dropout_flat)
        with tf.device("/cpu:0"):
            dropout_inputs = self._embedding_fn(dropout_inputs_ids)
        base_shape = array_ops.shape(base_next_inputs)
        return (array_ops.scatter_nd(indices=where_keeping,
                                     updates=keeping_inputs,
                                     shape=base_shape)
                + array_ops.scatter_nd(indices=where_dropout,
                                       updates=dropout_inputs,
                                       shape=base_shape))

      # if all elements in batch have reached to the ends of sequences,
      # stop considering word dropout, just feed zeros
      all_finished = math_ops.reduce_all(finished)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: base_next_inputs, maybe_dropout)
      return (finished, next_inputs, state)
