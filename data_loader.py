# -*- coding: utf-8 -*-
"""Functions and Class for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import copy

import tensorflow as tf

EOS_TOKEN = u"<eos>"
DROP_TOKEN = u"<drop>"
UNK_TOKEN = u"<unk>"
PAD_TOKEN = u"<pad>"

PAD_ID = 0
EOS_ID = 1
DROP_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, DROP_TOKEN, UNK_TOKEN]


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").split()


def _read_lines(filename):
    lines = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            words = line.replace("\n",EOS_TOKEN).split()
            lines.append(words)
    return lines


def _build_vocab(filename, max_vocab_size=None):
    # 10001 vocabulary size when maximum number is unset
    data = _read_words(filename)
    counter = collections.Counter(data)
    words, _ = zip(*sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    id_to_word = SPECIAL_TOKENS + list(words)
    if max_vocab_size:
        if len(id_to_word) < max_vocab_size:
            raise ValueError("max_vocab_size should be larger than "
                             "total vocabulary size of training dataset.")
            id_to_word = id_to_word[:max_vocab_size]
    # word_to_id = dict(zip(id_to_word, range(len(id_to_word))))
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    # if the words appear more often, id will be lower
    return word_to_id, id_to_word


def _append_pads(line, max_sequence_length):
    num_pads = max_sequence_length - len(line)
    assert(num_pads >= 0)
    line.extend([PAD_ID for _ in range(num_pads)])
    return line


def _line_to_word_ids_with_unkowns(line, word_to_id):
    return [word_to_id[word] if word in word_to_id else word_to_id[UNK_TOKEN]
                                                     for word in line]


def _reverse_sequence(sequence):
    sequence_clone = copy.deepcopy(sequence)
    sequence_clone.reverse()
    return sequence_clone


def _file_to_word_ids(filename, word_to_id):
    lines = _read_lines(filename)
    sequence = []
    sequence_rev = [] # reversed sequence (for encoder input)
    sequence_length = [] # length for each sequence
    max_sequence_length = max(len(line) for line in lines)
    for line in lines:
        sequence_length.append(len(line)) # including <eos> token
        words_ids = _line_to_word_ids_with_unkowns(line, word_to_id)
        words_ids_rev = _reverse_sequence(words_ids)
        words_ids_rev.pop(0) # remove <eos>
        #import pdb; pdb.set_trace()
        words_ids = _append_pads(words_ids, max_sequence_length)
        words_ids_rev = _append_pads(words_ids_rev, max_sequence_length)
        sequence.append(words_ids)
        sequence_rev.append(words_ids_rev)
    # len(training sequence) = 42068
    return sequence, sequence_rev, sequence_length


def get_raw_data_from_file(data_path=None, max_vocab_size=None):
    """
    Args:
        data_path: string path to the directory where simple-examples.tgz has
            been extracted.
    Returns:
        tuple (train_data, valid_data, test_data, word_to_id, id_to_word)
        where each of the data objects can be passed to PTBIterator.
    """
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")

    word_to_id, id_to_word = _build_vocab(train_path, max_vocab_size)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, word_to_id, id_to_word


class InputProducer(object):

    def __init__(self, data, word_to_id, id_to_word, config, name=None):
        # import ipdb; ipdb.set_trace()
        self.batch_size = config.batch_size
        self.sequence = data.q
        self.sequence_reversed = data.q_rev
        self.sequence_length = data.q_len
        self.sequence_number = len(self.sequence) # height
        self.seq_max_length = len(self.sequence[0]) # width 어차피 padding했음
        self.answer = data.a
        self.name = name
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_num = len(id_to_word)

        inputs = self.input_producer(self.sequence,
                                     self.sequence_reversed,
                                     self.sequence_length,
                                     self.answer,
                                     self.sequence_number,
                                     self.batch_size,
                                     self.name)
        self.x_enc, self.x_dec, self.y_dec = inputs[:3]
        self.len_enc, self.len_dec, self.answ_disc = inputs[3:]

    def input_producer(self, sequence, sequence_reversed, sequence_length,
                       answer, sequence_number, batch_size, name=None):
        batch_num = sequence_number // batch_size
        sequence_length_minus_one = [x-1 for x in sequence_length]
        with tf.name_scope(name, "InputProducer"):
            print("[*] converting data lists to tensors..")
            tf_sequence = tf.convert_to_tensor(sequence,
                                               name="sequence",
                                               dtype=tf.int32)
            tf_sequence_rev = tf.convert_to_tensor(sequence_reversed,
                                                   name="sequence_reversed",
                                                   dtype=tf.int32)
            tf_length_encoder = tf.convert_to_tensor(sequence_length_minus_one,
                                                     name="seq_length_encoder",
                                                     dtype=tf.int32)
            tf_length_decoder = tf.convert_to_tensor(sequence_length,
                                                     name="seq_length_decoder",
                                                     dtype=tf.int32)
            tf_answer = tf.convert_to_tensor(answer,
                                             name="sequence",
                                             dtype=tf.int32)
            print("[*] done!")
            i = tf.train.range_input_producer(batch_num, shuffle=False).dequeue()

            x_encoder = tf_sequence_rev[i*batch_size:(i+1)*batch_size]
            y_decoder = tf_sequence[i*batch_size:(i+1)*batch_size]
            length_encoder = tf_length_encoder[i*batch_size:(i+1)*batch_size]
            length_decoder = tf_length_decoder[i*batch_size:(i+1)*batch_size]
            answer_disc = tf_answer[i*batch_size:(i+1)*batch_size]

            eos = tf.expand_dims(tf.constant([EOS_ID]*batch_size),1)
            x_decoder = tf.concat([eos, y_decoder],1)

            x_encoder.set_shape([batch_size, None])
            x_decoder.set_shape([batch_size, None])
            y_decoder.set_shape([batch_size, None])
            length_encoder.set_shape([batch_size])
            length_decoder.set_shape([batch_size])
            answer_disc.set_shape([batch_size, 1])

        return (x_encoder, x_decoder, y_decoder, length_encoder,
                length_decoder, answer_disc)
