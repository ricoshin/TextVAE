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

class DecoderTrainingHelper(TrainingHelper):
  """A training helper that adds word dropout."""

  def __init__(self, inputs, sequence_length, embedding, teacher_forcing_prob,
               dropout_keep_prob, drop_token_id, time_major=False,
               scheduling_seed=None, dropout_seed=None, name=None):
    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (lambda ids: embedding_ops.embedding_lookup(
                                                                embedding, ids))
    self._teacher_forcing_prob = ops.convert_to_tensor(
      dropout_keep_prob, name="teacher_forcing_prob")
    self._dropout_keep_prob = ops.convert_to_tensor(
      dropout_keep_prob, name="dropout_keep_prob")
    if self._dropout_keep_prob.get_shape().ndims not in (0, 1):
      raise ValueError(
        "dropout_keep_prob must be either a scalar or a vector. "
        "saw shape: %s" % (self._dropout_keep_prob.get_shape()))
    self._drop_token_id = drop_token_id
    self._dropout_seed = dropout_seed
    self._scheduling_seed = scheduling_seed

    super(DecoderTrainingHelper, self).__init__(
      inputs=inputs,
      sequence_length=sequence_length,
      time_major=time_major,
      name=name)

  def initialize(self, name=None):
    return super(DecoderTrainingHelper, self).initialize(name=name)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    sample_ids = math_ops.cast(
        math_ops.argmax(outputs, axis=-1), dtypes.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    (finished, base_next_inputs, state) = (
        super(DecoderTrainingHelper, self).next_inputs(time=time,
                                                       outputs=outputs,
                                                       state=state,
                                                       sample_ids=sample_ids,
                                                       name=name))
    def maybe_sample_or_dropout():

      def random_replace(base_next_inputs, sub_next_inputs, keep_prob):
        if keep_prob==1.0: return base_next_inputs

        select_sample_noise = random_ops.random_uniform(
                                  [self.batch_size], seed=self._dropout_seed)
        replace = (keep_prob <= select_sample_noise) # true/false
        not_replace = (keep_prob > select_sample_noise)
        where_replace = math_ops.cast(array_ops.where(replace), dtypes.int32)
        where_not_replace = math_ops.cast(array_ops.where(not_replace),
                                          dtypes.int32)
        where_replace_flat = array_ops.reshape(where_replace, [-1])
        where_not_replace_flat = array_ops.reshape(where_not_replace, [-1])
        substitue_inputs = array_ops.gather(sub_next_inputs, where_replace_flat)
        substitue_inputs = self._embedding_fn(substitue_inputs)
        base_inputs = array_ops.gather(base_next_inputs, where_not_replace_flat)
        base_shape = array_ops.shape(base_next_inputs)
        return (array_ops.scatter_nd(indices=where_replace,
                                     updates=substitue_inputs,
                                     shape=base_shape)
              + array_ops.scatter_nd(indices=where_not_replace,
                                     updates=base_inputs,
                                     shape=base_shape))
      # perform scheduled argmax sampling
      next_inputs = random_replace(base_next_inputs=base_next_inputs,
                                   sub_next_inputs=sample_ids,
                                   keep_prob=self._teacher_forcing_prob)
      # perform word dropout
      drop_ids = array_ops.tile([self._drop_token_id], [self.batch_size])
      next_inputs = random_replace(base_next_inputs=next_inputs,
                                   sub_next_inputs=drop_ids,
                                   keep_prob=self._dropout_keep_prob)
      return next_inputs

    # if all elements in batch have reached to the ends of sequences,
    # stop considering word dropout, just feed zeros
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished, lambda: base_next_inputs, maybe_sample_or_dropout)
    return (finished, next_inputs, state)
