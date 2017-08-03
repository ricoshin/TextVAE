import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import SampleEmbeddingHelper
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell
from tensorflow.contrib.framework import get_or_create_global_step
from tensorflow.python.ops.nn import embedding_lookup, dynamic_rnn
from tensorflow.python.layers.core import Dense
from decoder_helper import WordDropoutTrainingHelper
from data_loader import DROP_ID, EOS_ID

class VariationalAutoencoder(object):

    def __init__(self, input_producer, embed_mat, config, is_train):

        with tf.variable_scope("VAE") as var_scope:
            x_enc = input_producer.x_enc
            x_dec = input_producer.x_dec
            y_dec = input_producer.y_dec
            len_enc = input_producer.len_enc
            len_dec = input_producer.len_dec

            max_len = input_producer.seq_max_length
            vocab_num = input_producer.vocab_num
            batch_size = config.batch_size
            hidden_size = config.hidden_size
            embed_dim = config.embed_dim

            is_GRU = config.is_GRU
            is_argmax_sampling = config.is_argmax_sampling
            word_keep_prob = config.word_dropout_keep_prob
            max_grad_norm = config.max_grad_norm
            learning_rate = config.learning_rate

            self.KL_weight = tf.Variable(0.0, "KL_weight")
            self.input_ids = y_dec

            def _lstm_cell():
                return BasicLSTMCell(num_units=hidden_size,
                                     forget_bias=1.0,
                                     state_is_tuple=True,
                                     reuse=tf.get_variable_scope().reuse)
            def _gru_cell():
                return GRUCell(num_units=hidden_size,
                               reuse=tf.get_variable_scope().reuse)

            cell = _gru_cell if is_GRU else _lstm_cell
            self.initial_state = cell().zero_state(batch_size, tf.float32)


            # encoder
            with tf.device("/cpu:0"):
                embed_init = tf.constant_initializer(embed_mat)\
                                if (embed_mat is not None) else None
                embedding = tf.get_variable("embedding", [vocab_num, embed_dim],
                                             initializer=embed_init,
                                             trainable=True)
                in_enc = embedding_lookup(embedding, x_enc)



            with tf.variable_scope("encoder"):
                out_tuple = dynamic_rnn(cell=cell(),
                                        inputs=in_enc,
                                        sequence_length=len_enc,
                                        initial_state=self.initial_state)
                (_, encoder_hidden) = out_tuple

                # linear layers for mu and log(var)
                latent_dim = hidden_size # may have to change this later
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
                stddev = tf.exp(0.5 * logvar) # standard deviation
                self.z = mu + tf.multiply(stddev, epsilon)

            # decoder
            with tf.device("/cpu:0"):
                in_dec = embedding_lookup(embedding, x_dec)

            with tf.variable_scope("decoder"):
                if is_train: # for training
                    helper = WordDropoutTrainingHelper(
                                               inputs=in_dec,
                                               sequence_length=len_dec,
                                               embedding=embedding,
                                               dropout_keep_prob=word_keep_prob,
                                               drop_token_id=DROP_ID)
                else : # for sampling
                    SamplingHelper = (GreedyEmbeddingHelper \
                        if is_argmax_sampling else SampleEmbeddingHelper)
                    start_tokens = tf.tile([EOS_ID], [batch_size])

                    helper = SamplingHelper(embedding=embedding,
                                            start_tokens=start_tokens,
                                            end_token=EOS_ID)
                # projection layer
                output_layer = Dense(units=vocab_num,
                                     activation=None,
                                     use_bias=True,
                                     trainable=True)

                # decoder
                decoder = BasicDecoder(cell=cell(),
                                       helper=helper,
                                       initial_state=self.z,
                                       output_layer=output_layer)

                # dynamic_decode
                out_tuple = dynamic_decode(decoder=decoder,
                                           output_time_major=False, #  speed
                                           impute_finished=True)

            # get all the variables in this scope
            self.vars = tf.contrib.framework.get_variables(var_scope)

        # (ouputs, state, sequence_length)
        (self.outputs, _, _) = out_tuple # final

        # (cell_outputs, sample_ids)
        (self.cell_outputs, self.sampled_ids) = self.outputs

        # compute softmax loss (reconstruction)
        len_out = tf.reduce_max(len_dec)
        targets = y_dec[:,:len_out]
        softmax_loss = sequence_loss(logits=self.cell_outputs,
                                     targets=targets,
                                     weights=tf.ones([batch_size, len_out]),
                                     average_across_timesteps=False,
                                     average_across_batch=True)

        self.AE_loss = tf.reduce_sum(softmax_loss)
        self.AE_loss_mean = tf.reduce_mean(softmax_loss)

        # compute KL loss (regularization)
        KL_term = 1 + logvar - tf.pow(mu, 2) - tf.exp(logvar)
        self.KL_loss = -0.5 * tf.reduce_sum(KL_term, reduction_indices=1)
        self.KL_loss_mean = tf.reduce_mean(self.KL_loss)

        # total loss
        self.loss = self.AE_loss + self.KL_weight * self.KL_loss_mean

        # optimization
        self.lr = tf.Variable(learning_rate, trainable=False, name="lr")

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.vars),
                                          max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.global_step = get_or_create_global_step()
        self.train_op = optimizer.apply_gradients(zip(grads, self.vars),
                                                  global_step=self.global_step)

        # learning_rate update
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
        self.lr_update = tf.assign(self.lr, self.new_lr)

        # KL weight update
        self.new_KL_weight = tf.placeholder(tf.float32, shape=[], name="new_kl")
        self.KL_weight_update = tf.assign(self.KL_weight, self.new_KL_weight)

        # summaries
        tf.summary.scalar("Loss/AE_mean", self.AE_loss_mean)
        tf.summary.scalar("Loss/KL_mean", self.KL_loss_mean)
        tf.summary.scalar("Loss/Total", self.AE_loss_mean + self.KL_loss_mean)
        tf.summary.scalar("Misc/KL_weight", self.KL_weight)
        tf.summary.scalar("Misc/mu_mean", tf.reduce_mean(mu))
        tf.summary.scalar("Misc/sigma_mean", tf.reduce_mean(stddev))
        tf.summary.scalar("Misc/learning_rate", self.lr)
        self.summary_op = tf.summary.merge_all()
        # print('end-of-function')

    # assign new learning rate
    def assign_lr(self, sess, new_lr):
        sess.run(self.lr_update, feed_dict={self.new_lr: new_lr})
        print("[INFO] learning rate updated")

    def assign_kl_weight(self, sess, weight):
        sess.run(self.KL_weight_update, feed_dict={self.new_KL_weight: weight})

    # this function is for debuging (remove later)
def sess():
    sv = tf.train.Supervisor()
    sess = sv.PrepareSession()
    #import pdb; pdb.set_trace()
    return sess
