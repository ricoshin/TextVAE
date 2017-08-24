from __future__ import print_function

import tensorflow as tf
from data_loader import InputProducer
from data_loader import get_raw_data_from_file
from ctrl_vae_model import CtrlVAEModel
from data_loader import PAD_ID, EOS_ID
from tqdm import tqdm
from wordvec import load_glove_embeddings
import numpy as np
import os

from data import load_simple_questions_dataset

FLAGS = tf.app.flags.FLAGS

class CtrlVAETrainer(object):
    def __init__(self, config):
        self.config = config

        sq_dataset = load_simple_questions_dataset(config)
        train_data, valid_data, embed_mat, self.word_to_id = sq_dataset
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

        # Generate input
        train_input = InputProducer(data=train_data, word_to_id=self.word_to_id,
                                    id_to_word=self.id_to_word, config=config)

        # Build model
        self.model = CtrlVAEModel(input_producer= train_input,
                                  embed_mat=embed_mat,
                                  config=config,
                                  is_train=FLAGS.is_train)

        # Supervisor & Session
        self.sv = tf.train.Supervisor(logdir=FLAGS.model_subdir,
                                      save_model_secs=config.save_model_secs)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)

        self.sess = self.sv.PrepareSession(config=sess_config)

    def train2(self):
        self.sess.run(self.model.test)
        import ipdb; ipdb.set_trace()
        print("dd")

    def train(self):
        # Initialize progress step
        progress_bar = tqdm(total=self.config.max_step)
        step = self.sess.run(self.model.global_step)
        progress_bar.update(step)
        is_print = (lambda step : step % self.config.print_step == 0)

        # Main loop
        while not(self.sv.should_stop()):
            progress_bar.update(1)
            step += 1
            if step > self.config.max_step:
                self.sv.request_stop()

            # update KL anealing weight
            from math import cos, pi
            aneal_step = self.config.kl_anealing_step
            if step < aneal_step:
                new_kl_weight = (-cos(pi*step/aneal_step)+1)/2
                self.model.assign_kl_weight(self.sess, new_kl_weight)

            aneal_step = self.config.gen_lr_anealing_step
            max_lr = self.config.learning_rate
            if step < aneal_step:
                new_gen_lr = (-cos(pi*step/aneal_step)+1)/2*max_lr
                self.model.assign_gen_lr(self.sess, new_gen_lr)


            fetches = {"vae_train" : self.model.vae_train,
                       "gen_train" : self.model.gen_train,
                       "dis_train" : self.model.dis_train}

            if is_print(step):
                fetches.update({"summary_op" : self.model.summary_op,
                                "input_ids" : self.model.input_ids,
                                "answer" : self.model.answer,
                                "vae_sample" : self.model.vae_sample,
                                "gen_sample" : self.model.gen_sample,
                                "gen_c_sample" : self.model.gen_c_sample,
                                "dis_sample" : self.model.dis_sample,
                                "ae_loss" : self.model.ae_loss_mean,
                                "kl_loss" : self.model.kl_loss_mean,
                                "kl_weight" : self.model.kl_weight,
                                "gen_lr" : self.model.gen_lr})

            result = self.sess.run(fetches)

            if is_print(step):
                print("[*] AE_loss: {} / KL_loss: {} / KL weight : {}"
                      " / Generator learning rate : {}"
                      "".format(result["ae_loss"], result["kl_loss"],
                                result['kl_weight'], result['gen_lr']))

                self._print_results(result, self.id_to_word, 30)
                #import pdb; pdb.set_trace()
                #raw_input("Press Enter to continue...")

    def _print_results(self, result, id_to_word, max_words):
        self._print_asterisk()

        def batch_to_str(result_batch):
            str_batch = list()
            for result in result_batch:
                words = self._ids_to_words(result, id_to_word)
                str_batch.append(self._words_to_str(words, max_words))
            return str_batch

        vae_in = batch_to_str(result['input_ids'])
        vae_out = batch_to_str(result['vae_sample'])
        gen_in = batch_to_str(result['answer'])
        gen_out = batch_to_str(result['gen_sample'])
        gen_pred = batch_to_str([[res] for res in result['gen_c_sample']])
        dis_out = batch_to_str([[res] for res in result['dis_sample']])

        print('\n## VAE ##')
        print("[Q_0] " + vae_in[0] + "\n[Q_hat_0] " + vae_out[0])
        print("[Q_1] " + vae_in[1] + "\n[Q_hat_1] " + vae_out[1])
        print("[Q_2] " + vae_in[2] + "\n[Q_hat_2] " + vae_out[2])
        print("\n## Generator ##")
        print("[A_cond_0] " + gen_in[0] + "/ [A_pred_0] " + gen_pred[0])
        print("[Q_sampled_0] " + gen_out[0])
        print("[A_cond_1] " + gen_in[1] + "/ [A_pred_1] " + gen_pred[1])
        print("[Q_sampled_1] " + gen_out[1])
        print("[A_cond_2] " + gen_in[2] + "/ [A_pred_2] " + gen_pred[2])
        print("[Q_sampled_2] " + gen_out[2])
        print("\n## Discriminator ##")
        print("[Q_0] " + vae_in[0])
        print("[A_true_0] " + gen_in[0] + " / [A_pred_0] " + dis_out[0])
        print("[Q_1] " + vae_in[1])
        print("[A_true_1] " + gen_in[1] + " / [A_pred_1] " + dis_out[1])
        print("[Q_2] " + vae_in[2])
        print("[A_true_2] " + gen_in[2] + " / [A_pred_2] " + dis_out[2])
        self._print_asterisk()

    def sample(self):
        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        interval_num = 19
        key = ''
        self._print_asterisk()

        while(key != 'q'):

            answer_id = np.squeeze(self.sess.run(self.model.answer))[0]
            answer_str = self.id_to_word[answer_id]
            print("Sampling conditioned on the answer : " + answer_str)
            answer = np.expand_dims([answer_id]*batch_size, 1)
            feeds = {self.model.answer : answer}

            if self.config.is_interpolation:
                print("Walking in the latent space...")
                z_a = np.random.normal(0, 1, (1, hidden_size))
                z_b = np.random.normal(0, 1, (1, hidden_size))
                diff = (z_b - z_a) / interval_num
                intervals = [z_a + diff*i if i<=interval_num\
                                          else np.tile([0], hidden_size)\
                                              for i in range(batch_size)]
                vae_z = np.vstack(intervals)
                feeds.update({self.model.gen_z : vae_z})

            gen_sample = self.sess.run(self.model.gen_sample, feeds)

            for i, ids in enumerate(gen_sample):
                if i > interval_num: break
                words = self._ids_to_words(ids, self.id_to_word)
                print(self._words_to_str(words, max_words=40))

            key = raw_input("Press any key to continue('q' to quit)...")
            self._print_asterisk()
        self.sv.request_stop()

    def interpolate_samples(self):
        interval_num = 9
        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        key = ''

        while(key != 'q'):
            z_a = np.random.normal(0, 1, (1, hidden_size))
            z_b = np.random.normal(0, 1, (1, hidden_size))
            diff = (z_b - z_a) / interval_num
            intervals = [z_a + diff*i if i<=interval_num\
                                      else np.tile([0], hidden_size)\
                                           for i in range(batch_size)]
            z_batch = np.vstack(intervals)
            sampled_ids = self.sess.run(self.VAE.sampled_ids, {self.VAE.z: z_batch})
            self._print_asterisk()
            for i in range(interval_num+1):
                words = self._ids_to_words(sampled_ids[i], self.id_to_word)
                sentence_str = self._words_to_str(words, 40)
                print('[{}] '.format(i) + sentence_str)
                self._print_asterisk()

            key = raw_input("Press any key to continue('q' to quit)...")
            if key == 'q': break
        self.sv.request_stop()

    def _print_asterisk(self):
        print("*"*120)

    def _ids_to_words(self, word_ids, id_to_word):
        return [id_to_word[word_id]
                    for word_id in word_ids if word_id!=PAD_ID]
                    # and word_id!=EOS_ID)]'

    def _words_to_str(self, words, max_words):
        # if sentence is too long,
        # cut it short and punctuate with ellipsis(...)
        if len(words) > max_words:
            words[:max_words].append("...")
        return " ".join(words)

    def _print_reconst_samples(self, in_ids, out_ids, id_to_word, answer,
                              sentence_num, max_words):
        self._print_asterisk()

        def ids_to_str(word_ids):
            words = self._ids_to_words(word_ids, id_to_word)
            return self._words_to_str(words, max_words)

        for i in range(sentence_num):
            in_str = ids_to_str(in_ids[i])
            out_str = ids_to_str(out_ids[i])
            answ_exp_str = ids_to_str(answer['label'][i])
            answ_act_str = ids_to_str([answer['pred'][i]])

            print("[X] " + in_str + "\n" + "[Y] " + out_str)
            print('[A] Expected:', answ_exp_str, '/', 'Actual:', answ_act_str)
            self._print_asterisk()
