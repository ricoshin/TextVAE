import numpy as np
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
from ctrl_vae_helper import CtrlVAEModelingHelper

ALMOST_ZERO = 1e-12
  
class CtrlVAEModel(object):

    def __init__(self, input_producer, embed_mat, config, is_train):
        x_enc = input_producer.x_enc
        x_dec = input_producer.x_dec
        y_dec = input_producer.y_dec
        len_enc = input_producer.len_enc
        len_dec = input_producer.len_dec
        self.answer = input_producer.answ_disc
        self.answer_length = input_producer.answ_len_disc
        self.answer_max_length = input_producer.answ_max_length

        max_len = input_producer.seq_max_length
        vocab_num = input_producer.vocab_num
        config.update(**dict(max_len=max_len, vocab_num=vocab_num))
        # import ipdb; ipdb.set_trace()
        self.kl_weight = tf.Variable(0.0, "KL_weight")
        self.input_ids = y_dec

        modeler = CtrlVAEModelingHelper(config, embed_mat)

        with tf.variable_scope("CtrlVAE"):

            ### VAE ############################################################

            # encoder
            x_enc_onehot = tf.one_hot(x_enc, vocab_num)
            out_tuple = modeler.encoder(x_enc_onehot=x_enc_onehot,
                                        len_enc=len_enc)
            (vae_z, vae_mu, vae_logvar) = out_tuple

            # holistic representation
            # with tf.device("/cpu:0"):
            #    vae_c = embedding_lookup(modeler.embed, self.answer)
            with tf.variable_scope('vae_c'):
                vae_c = encode_answer(config, modeler.embed, self.answer, self.answer_length)
            # vae_c = tf.reshape(vae_c, [config.batch_size, -1])
            vae_represent = tf.concat([vae_z, vae_c], axis=1)

            # decoder
            x_dec_onehot = tf.one_hot(x_dec, config.vocab_num)
            out_tuple = modeler.decoder(initial_state=vae_represent,
                                        x_dec_onehot=x_dec_onehot,
                                        len_dec=len_dec,
                                        is_teacher_forcing=True)

            (vae_outputs, vae_state, vae_outputs_len) = out_tuple # final
            (self.vae_output, self.vae_sample) = vae_outputs

            ### Generator ######################################################

            # random z and c from the prior
            self.gen_z = tf.random_normal([config.batch_size,
                                           config.hidden_size])
            self.gen_c = vae_c
            gen_represent = tf.concat([self.gen_z, self.gen_c], axis=1)

            # generator (decoder)
            x_dec_onehot = tf.one_hot(x_dec, config.vocab_num)
            out_tuple = modeler.decoder(initial_state=gen_represent,
                                        x_dec_onehot=x_dec_onehot,
                                        len_dec=len_dec,
                                        is_teacher_forcing=True,
                                        reuse=True)

            (gen_outputs, gen_state, gen_outputs_len) = out_tuple # final
            (self.gen_output, self.gen_sample) = gen_outputs
            gen_outputs_onehot = softmax(self.gen_output/ALMOST_ZERO)

            # discriminator (for c code)
            decoder = lambda inital_state: decode_answer(config=config, 
                                            max_length=self.answer_max_length,
                                            embed=modeler.embed,
                                            initial_state=inital_state,
                                            inputs=self.answer,
                                            inputs_length=self.answer_length,
                                            is_train=is_train)  
            out_tuple = modeler.discriminator(inputs=gen_outputs_onehot,
                                              inputs_length=gen_outputs_len,
                                              decoder=decoder)
            (self.gen_c_output, self.gen_c_sample) = out_tuple

            # encoder again (for z code ; additional discriminator)
            out_tuple = modeler.encoder(x_enc_onehot=gen_outputs_onehot,
                                        len_enc=gen_outputs_len,
                                        reuse=True)
            (gen_z, dis_mu, dis_logvar) = out_tuple

            ### Discriminator ##################################################

            # discriminator (for training)
            x_dis_onehot = tf.one_hot(x_enc, config.vocab_num)
            decoder = lambda inital_state: decode_answer(config=config, 
                                            max_length=self.answer_max_length,
                                            embed=modeler.embed,
                                            initial_state=inital_state,
                                            inputs=self.answer,
                                            inputs_length=self.answer_length,
                                            is_train=is_train)  
            out_tuple = modeler.discriminator(inputs=x_dis_onehot,
                                              inputs_length=gen_outputs_len,
                                              reuse=True,
                                              decoder=decoder)
            (self.dis_outputs, self.dis_sample) = out_tuple
            
        ########################################################################
        # get all the variables in this scope
        self.vars = get_variables("CtrlVAE")
        self.enc_vars = get_variables("CtrlVAE/encoder")
        self.gen_vars = get_variables("CtrlVAE/decoder")
        self.dis_vars = get_variables("CtrlVAE/discriminator")
        self.vae_vars = self.enc_vars + self.gen_vars
        ########################################################################
        # compute AE loss (reconstruction)
        len_out = tf.reduce_max(vae_outputs_len)
        targets = y_dec[:,:len_out]
        weights = tf.sequence_mask(vae_outputs_len, dtype=tf.float32)

        softmax_loss = sequence_loss(logits=self.vae_output,
                                     targets=targets,
                                     weights=weights,
                                     average_across_timesteps=True,
                                     average_across_batch=True)

        self.ae_loss = self.ae_loss_mean = softmax_loss
        #self.ae_loss_mean = tf.reduce_mean(softmax_loss)

        # compute KL loss (regularization)
        KL_term = 1 + vae_logvar - tf.pow(vae_mu, 2) - tf.exp(vae_logvar)
        self.kl_loss = -0.5 * tf.reduce_sum(KL_term, reduction_indices=1)
        self.kl_loss_mean = tf.reduce_mean(self.kl_loss)

        # VAE total loss
        self.vae_loss = self.ae_loss + self.kl_weight * self.kl_loss_mean
        ########################################################################
        # c code loss
        answer_labels = tf.one_hot(self.answer, config.vocab_num)

        c_loss = softmax_cross_entropy_with_logits(labels=answer_labels,
                                                   logits=self.gen_c_output)
        self.c_loss = tf.reduce_mean(c_loss)

        # z code loss
        mu_loss = mean_pairwise_squared_error(vae_mu, dis_mu)
        logvar_loss = mean_pairwise_squared_error(vae_logvar, dis_logvar)
        self.z_loss = (mu_loss + logvar_loss) / 2

        # generator total loss
        self.gen_loss = self.c_loss + self.z_loss
        ########################################################################
        # discriminator training loss
        dis_loss = softmax_cross_entropy_with_logits(labels=answer_labels,
                                                      logits=self.dis_outputs)
        self.dis_loss = tf.reduce_mean(dis_loss)
        ########################################################################

        # optimization
        lr = config.learning_rate
        self.vae_lr = tf.Variable(lr, trainable=False, name="vae_lr")
        self.gen_lr = tf.Variable(0.0, trainable=False, name="gen_lr")
        self.dis_lr = tf.Variable(lr, trainable=False, name="dis_lr")

        vae_optim = tf.train.AdamOptimizer(self.vae_lr)
        gen_optim = tf.train.AdamOptimizer(self.gen_lr)
        dis_optim = tf.train.AdamOptimizer(self.dis_lr)

        vae_grads = tf.gradients(self.vae_loss, self.vae_vars)
        gen_grads = tf.gradients(self.gen_loss, self.gen_vars)
        dis_grads = tf.gradients(self.dis_loss, self.dis_vars)

        vae_grads, _ = tf.clip_by_global_norm(vae_grads, config.max_grad_norm)
        gen_grads, _ = tf.clip_by_global_norm(gen_grads, config.max_grad_norm)
        dis_grads, _ = tf.clip_by_global_norm(dis_grads, config.max_grad_norm)

        self.global_step = get_or_create_global_step()
        self.vae_train = vae_optim.apply_gradients(zip(vae_grads,self.vae_vars))
        self.gen_train = gen_optim.apply_gradients(zip(gen_grads,self.gen_vars))
        self.dis_train = dis_optim.apply_gradients(
                              zip(dis_grads,self.dis_vars), self.global_step)

        # learning_rate update
        self.new_gen_lr = tf.placeholder(tf.float32,shape=[],name="new_gen_lr")
        self.gen_lr_update = tf.assign(self.gen_lr, self.new_gen_lr)

        # KL weight update
        self.new_kl_weight = tf.placeholder(tf.float32, shape=[], name="new_kl")
        self.kl_weight_update = tf.assign(self.kl_weight, self.new_kl_weight)

        # summaries
        tf.summary.scalar("Loss/ae_mean", self.ae_loss_mean)
        tf.summary.scalar("Loss/kl_mean", self.kl_loss_mean)
        tf.summary.scalar("Loss/Total", self.ae_loss_mean + self.kl_loss_mean)
        tf.summary.scalar("Misc/kl_weight", self.kl_weight)
        tf.summary.scalar("Misc/mu_mean", tf.reduce_mean(vae_mu))
        tf.summary.scalar("Misc/logvar_mean", tf.reduce_mean(vae_logvar))
        tf.summary.scalar("Misc/gen_lr", self.gen_lr)
        self.summary_op = tf.summary.merge_all()
        # print('end-of-function')

    # assign new learning rate
    def assign_gen_lr(self, sess, new_gen_lr):
        sess.run(self.gen_lr_update, feed_dict={self.new_gen_lr: new_gen_lr})

    def assign_kl_weight(self, sess, weight):
        sess.run(self.kl_weight_update, feed_dict={self.new_kl_weight: weight})

# this function is for debuging (remove later)
def sess():
    sv = tf.train.Supervisor()
    sess = sv.PrepareSession()
    #import pdb; pdb.set_trace()
    return sess

def encode_answer(config, embed, inputs, inputs_length):
    # inputs=[batch_size, max_answ_len]

    cell = GRUCell(num_units=config.hidden_size)
    # inputs=[batch_size, max_answ_len, embed_dim(50)]
    with tf.device("/cpu:0"):
        inputs = embedding_lookup(embed, inputs)
    _, state = dynamic_rnn(cell=cell,
                           inputs=inputs,
                           sequence_length=inputs_length,
                           dtype=tf.float32)
    # state=[batch_size, hidden_size]

    W_answ = tf.get_variable('W_answ', [config.hidden_size, config.embed_dim])
    b_answ = tf.get_variable('b_answ', [config.embed_dim])

    # [batch_size, embed_dim]
    return tf.matmul(state, W_answ) + b_answ

def decode_answer(config, max_length, embed, initial_state, inputs, inputs_length, is_train=False):
    with tf.variable_scope('decoder') as scope:
        cell = GRUCell(num_units=config.hidden_size)
        # decoder
        dropout_keep_prob = config.word_dropout_keep_prob
        is_argmax_sampling = config.is_argmax_sampling

        initial_state = dense(inputs=initial_state,
                              units=config.hidden_size,
                              activation=None,
                              use_bias=True,
                              trainable=True)

        if is_train: # teacher forcing for training
            assert(dropout_keep_prob is not None)
            inputs = embedding_lookup(embed, inputs)
            helper = WordDropoutTrainingHelper(inputs=inputs,
                                               sequence_length=inputs_length,
                                               embedding=embed,
                                               dropout_keep_prob=dropout_keep_prob,
                                               drop_token_id=UNK_ID)
        else : # for sampling
            SamplingHelper = (GreedyEmbeddingHelper \
                if is_argmax_sampling else SampleEmbeddingHelper)
            start_tokens = tf.tile([EOS_ID], [config.batch_size])

            helper = SamplingHelper(embedding=embed,
                                    start_tokens=start_tokens,
                                    end_token=EOS_ID)
        # projection layer
        output_layer = Dense(units=config.vocab_num,
                             activation=None,
                             use_bias=True,
                             trainable=True)

        # decoder
        decoder = BasicDecoder(cell=cell,
                               helper=helper,
                               initial_state=initial_state,
                               output_layer=output_layer)

        # dynamic_decode
        out_tuple = dynamic_decode(decoder=decoder,
                                   output_time_major=False, # speed
                                   impute_finished=True)


        ((outputs, _), _, _) = out_tuple
        pad = np.zeros([config.batch_size, max_length])
        pad = tf.one_hot(pad, config.vocab_num)
        outputs = tf.concat([outputs, pad], axis=1)
        outputs = outputs[:, :max_length, :]
        return outputs
 
