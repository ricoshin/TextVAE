import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import SampleEmbeddingHelper
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.contrib.framework import get_or_create_global_step
from tensorflow.contrib.framework import get_variables
from tensorflow.python.ops.nn import embedding_lookup, dynamic_rnn
from tensorflow.python.ops.nn import softmax, softmax_cross_entropy_with_logits
from tensorflow.python.ops.losses.losses import mean_pairwise_squared_error
from tensorflow.python.layers.core import Dense, dense
from decoder_helper import WordDropoutTrainingHelper
from data_loader import UNK_ID, EOS_ID

class CtrlVAEModelingHelper(object):
    def __init__(self, config, embed_init=None):
        self.config = config

        def _lstm_cell(reuse=False):
            return BasicLSTMCell(num_units=config.hidden_size,
                                 forget_bias=1.0,
                                 state_is_tuple=True,
                                 reuse=reuse)
        def _gru_cell(reuse=False):
            return GRUCell(num_units=config.hidden_size,
                           reuse=reuse)

        self.cell = _gru_cell if config.is_GRU else _lstm_cell

        ### embedding ###
        embed_initializer = tf.constant_initializer(embed_init)\
                                if (embed_init is not None) else None

        with tf.device("/cpu:0"):
            self.embed = tf.get_variable("embedding",
                                         [config.vocab_num, config.embed_dim],
                                         initializer=embed_initializer,
                                         trainable=config.is_trainable_embed)

    def _soft_embedding_lookup(self, embed_mat, onehot_like):
        with tf.name_scope("soft_embedding_lookup"):
            batch_size = onehot_like.shape.as_list()[0] # [batch, max_len, vocab_num]
            vocab_num, embed_dim = embed_mat.shape.as_list() # [vocab_num, embed_dim]
            onehot_like = tf.reshape(onehot_like, [-1, vocab_num])
            looked_up = tf.matmul(onehot_like, embed_mat, a_is_sparse=True)
            looked_up = tf.reshape(looked_up, [batch_size, -1, embed_dim])
            return looked_up

    def encoder(self, x_enc_onehot, len_enc, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):

            in_enc = self._soft_embedding_lookup(self.embed, x_enc_onehot)
            initial_state = self.cell().zero_state(self.config.batch_size,
                                                   tf.float32)

            out_tuple = dynamic_rnn(cell=self.cell(reuse),
                                    inputs=in_enc,
                                    sequence_length=len_enc,
                                    initial_state=initial_state)
            (_, encoder_hidden) = out_tuple

            # linear layers for mu and log(var)
            latent_dim = hidden_size = self.config.hidden_size
            W_mu = tf.get_variable("W_mu", [hidden_size,latent_dim])
            b_mu = tf.get_variable("b_mu", [latent_dim])
            W_logvar = tf.get_variable("W_logvar", [hidden_size,latent_dim])
            b_logvar = tf.get_variable("b_logvar", [latent_dim])
            #l2_loss = tf.nn.l2_loss(W_mu) + tf.nn.l2_loss(W_logvar)

            mu = tf.matmul(encoder_hidden, W_mu) + b_mu
            logvar = tf.matmul(encoder_hidden, W_logvar) + b_logvar

            # sample epsilon
            epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')

            # sample latent variable
            stddev = tf.exp(0.5 * logvar) # standard
            z = mu + tf.multiply(stddev, epsilon)
            return z, mu, logvar

    def decoder(self, initial_state, x_dec_onehot, len_dec,
                is_teacher_forcing=False, reuse=False):
        # decoder
        with tf.variable_scope("decoder", reuse=reuse):
            dropout_keep_prob = self.config.word_dropout_keep_prob
            is_argmax_sampling = self.config.is_argmax_sampling
            in_dec = self._soft_embedding_lookup(self.embed, x_dec_onehot)

            initial_state = dense(inputs=initial_state,
                                  units=self.config.hidden_size,
                                  activation=None,
                                  use_bias=True,
                                  trainable=True,
                                  name='initial_layer')

            if is_teacher_forcing: # for training
                assert(dropout_keep_prob is not None)
                helper = WordDropoutTrainingHelper(
                                     inputs=in_dec,
                                     sequence_length=len_dec,
                                     embedding=self.embed,
                                     dropout_keep_prob=dropout_keep_prob,
                                     drop_token_id=UNK_ID)
            else : # for sampling
                SamplingHelper = (GreedyEmbeddingHelper \
                    if is_argmax_sampling else SampleEmbeddingHelper)
                start_tokens = tf.tile([EOS_ID], [self.config.batch_size])

                helper = SamplingHelper(embedding=self.embed,
                                        start_tokens=start_tokens,
                                        end_token=EOS_ID)
            # projection layer
            output_layer = Dense(units=self.config.vocab_num,
                                 activation=None,
                                 use_bias=True,
                                 trainable=True,
                                 name='output_layer')

            # decoder
            decoder = BasicDecoder(cell=self.cell(reuse),
                                   helper=helper,
                                   initial_state=initial_state,
                                   output_layer=output_layer)

            # dynamic_decode
            out_tuple = dynamic_decode(decoder=decoder,
                                       output_time_major=False, # speed
                                       impute_finished=True)
            return out_tuple

    def discriminator(self, inputs, inputs_length, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            inputs = self._soft_embedding_lookup(self.embed, inputs)
            _, state = dynamic_rnn(cell=self.cell(reuse),
                                   inputs=inputs,
                                   sequence_length=inputs_length,
                                   dtype=tf.float32)
            output_layer = Dense(self.config.vocab_num)
            outputs = output_layer(state)
            predicted = tf.argmax(outputs, 1)
            return outputs, predicted
