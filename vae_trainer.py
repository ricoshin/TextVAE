import tensorflow as tf
from data_loader import InputProducer
from data_loader import get_raw_data_from_file
from vae_model import VariationalAutoencoder
from data_loader import PAD_ID, EOS_ID
from tqdm import tqdm
from wordvec import load_glove_embeddings
import numpy as np
import os

from data import load_simple_questions_dataset

FLAGS = tf.app.flags.FLAGS

class VAETrainer(object):
    def __init__(self, config):
        self.config = config

        sq_dataset = load_simple_questions_dataset(config)
        train_data, valid_data, embed_mat, word_to_id = sq_dataset
        self.id_to_word = {i: w for w, i in word_to_id.items()}

        # Generate input
        train_input = InputProducer(data=train_data, word_to_id=word_to_id,
                                    id_to_word=self.id_to_word, config=config)

        # Build model
        self.VAE = VariationalAutoencoder(input_producer= train_input,
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


    def _init_ptb(self, config):

        self.config = config

        # Raw data load (PTB dataset)
        raw_data = get_raw_data_from_file(data_path=FLAGS.data_dir,
                                          max_vocab_size=config.max_vocab_size)

        train_data, valid_data, test_data, word_to_id, id_to_word = raw_data

        embed_file_path = os.path.join(FLAGS.data_dir, 'embed_matrix.npy')

        if os.path.exists(embed_file_path) and not config.reload_embed:
            print("[*] loading embedding matrix from 'embed_matrix.npy'...")
            embed_mat = np.load(embed_file_path)
            print("[*] done!")
        else:
            print("[*] initially building embedding matrix...")
            embed_mat = load_glove_embeddings(data_dir=FLAGS.glove_dir,
                                              num_tokens='42B',
                                              embed_dim=config.embed_dim,
                                              word_to_id=word_to_id)
            print("[*] and saving data as file...")
            np.save(embed_file_path, embed_mat)
            print("[*] done!")

        self.id_to_word = id_to_word

        # Generate input
        train_input = InputProducer(data=train_data, word_to_id=word_to_id,
                                    id_to_word=id_to_word, config=config)

        # Build model
        self.VAE = VariationalAutoencoder(input_producer= train_input,
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

    def train(self):
        # Initialize progress step
        progress_bar = tqdm(total=self.config.max_step)
        step = self.sess.run(self.VAE.global_step)
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
            if step < self.config.KL_anealing_step:
                new_kl_weight = (-cos(pi*step/self.config.KL_anealing_step)+1)/2
            else: new_kl_weight = 1
            self.VAE.assign_kl_weight(self.sess, new_kl_weight)

            feeds = {"train_op" : self.VAE.train_op}

            if is_print(step):
                feeds.update({"train_op" : self.VAE.train_op,
                              "summary_op" : self.VAE.summary_op,
                              "input_ids" : self.VAE.input_ids,
                              "sampled_ids" : self.VAE.sampled_ids,
                              "AE_loss" : self.VAE.AE_loss_mean,
                              "KL_loss" : self.VAE.KL_loss_mean,
                              "KL_weight" : self.VAE.KL_weight})

            result = self.sess.run(feeds)

            if is_print(step):
                print("[*] AE_loss: {} / KL_loss: {} / KL weight : {}".format(
                      result["AE_loss"],result["KL_loss"],result['KL_weight']))
                (in_ids, out_ids) = (result['input_ids'], result['sampled_ids'])
                self._print_reconst_samples(in_ids, out_ids, self.id_to_word,
                                            sentence_num=5, max_words=40)
                #import pdb; pdb.set_trace()
                #raw_input("Press Enter to continue...")

    def sample(self):
        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        key = ''
        self._print_asterisk()

        while(key != 'q'):
            z = np.random.normal(0, 1, (batch_size, hidden_size))
            sampled_ids = self.sess.run(self.VAE.sampled_ids, {self.VAE.z: z})

            for ids in sampled_ids:
                words = self._ids_to_words(ids, self.id_to_word)
                self._print_asterisk()
                print(self._words_to_str(words, max_words=40))
                self._print_asterisk()
                key = raw_input("Press any key to continue('q' to quit)...")
                if key == 'q': break
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

    def _print_reconst_samples(self, in_ids, out_ids, id_to_word,
                              sentence_num, max_words):
        self._print_asterisk()
        for i in range(sentence_num):
            # covert from ids to words using 'id_to_word' list
            in_words = self._ids_to_words(in_ids[i], id_to_word)
            out_words = self._ids_to_words(out_ids[i], id_to_word)

            in_str = self._words_to_str(in_words, max_words)
            out_str = self._words_to_str(out_words, max_words)

            print("[X] " + in_str + "\n" + "[Y] " + out_str)
            self._print_asterisk()
