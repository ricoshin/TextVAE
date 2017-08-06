import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.nn import (dynamic_rnn, embedding_lookup,
                                      softmax_cross_entropy_with_logits)
from tensorflow.python.layers.core import Dense


class Discriminator(object):
    def __init__(self, cfg, word_embd, max_ques_len, input_producer, generated=None):
        batch_size = cfg.batch_size
        vocab_size = len(word_embd)
        with tf.variable_scope('disc'):
            word_embd = tf.get_variable('word_embd',
                                        shape=word_embd.shape,
                                        initializer=tf.constant_initializer(word_embd))
            if generated:
                self.ques = generated['ques']
                self.ques_len = generated['ques_len']

                # soft embedding_lookup
                ques = tf.reshape(self.ques, [-1, vocab_size])
                ques = tf.matmul(ques, word_embd)
                ques = tf.reshape(ques, [batch_size, -1, cfg.embed_dim])
            else:
                self.ques = tf.placeholder(tf.int32,
                                           shape=[None, max_ques_len],
                                           name='question')
                self.ques_len = tf.placeholder(tf.int32,
                                               shape=[None],
                                               name='question_length')
                ques = embedding_lookup(word_embd, self.ques)
            self.answ = input_producer.answ_disc
            cell = GRUCell(cfg.hidden_size)
            _, state = dynamic_rnn(cell,
                                   ques,
                                   sequence_length=self.ques_len,
                                   dtype=tf.float32)
            output_layer = Dense(vocab_size)
            logits = output_layer(state)
            labels = tf.one_hot(self.answ, vocab_size)
            self.pred = tf.argmax(logits, 1)
            loss = softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
            self.loss = tf.reduce_mean(loss)
