import tensorflow as tf
from data_loader import InputProducer
from data_loader import get_raw_data_from_file
from vae_model import VariationalAutoencoder
from data_loader import PAD_ID, EOS_ID
from tqdm import tqdm
import numpy as np

FLAGS = tf.app.flags.FLAGS

class VAETrainer(object):
    def __init__(self, config):

        self.config = config
        # Raw data load (PTB dataset)
        raw_data = get_raw_data_from_file(FLAGS.data_dir, config.max_vocab_size)
        train_data, valid_data, test_data, word_to_id, id_to_word = raw_data
        self.id_to_word = id_to_word

        # Generate input
        train_input = InputProducer(train_data, word_to_id, id_to_word, config)

        # Build model
        self.VAE = VariationalAutoencoder(train_input, config, FLAGS.is_train)

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
                sv.request_stop()

            # for KL anealing weight update
            new_kl_weight = (np.tanh((step-2500)/1000)+1)/2
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
                self._print_ids_to_words(in_ids, out_ids, self.id_to_word,
                                         sentence_num=5, max_words=40)
                #import pdb; pdb.set_trace()
                #raw_input("Press Enter to continue...")

    def sample(self):
        self._print_asterisk()
        while(True):
            z = tf.random_normal([batch_size, self.VAE.hidden_size], name='z')
            sampled_ids = self.sess.run(self.VAE.sample_ids, {self.z : z})
            sampled_words = self._ids_to_words(sampled_ids, self.id_to_word)
            for words in sampled_words:
                print(self._words_to_str(sampled_ids, 40))
                self._print_asterisk()
                key = raw_input("Press any key to continue... or 'x' to quit")
                if key == 'x': break

    def _print_asterisk():
        print("*"*120)

    def _ids_to_words(self, word_ids, id_to_word):
        return [id_to_word[word_id]
                    for word_id in word_ids if (word_id!=PAD_ID)]
                    # and word_id!=EOS_ID)]'

    def _words_to_str(words, max_words):
        # if sentence is too long,
        # cut it short and punctuate with ellipsis(...)
        if len(words) > max_words:
            words[:max_words].append("...")
        return " ".join(words)

    def _print_recont_samples(self, in_ids, out_ids, id_to_word,
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
