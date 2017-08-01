import data_loader
from data_loader import InputProducer
from shutil import copyfile
from model_vae import VariationalAutoencoder
from data_loader import PAD_ID, EOS_ID
from tqdm import tqdm
import numpy as np

FLAGS = tf.app.flags.FLAGS

class VAETrainer(object):
    def __init__(self, config):

        # Generate ConfigStruct instance
        config = ConfigStruct(**configs["model"])
        config.update(**configs["train"])

        # Raw data load (PTB dataset)
        raw_data = data_loader.get_raw_data_from_file(FLAGS.data_dir,
                                                      config.max_vocab_size)
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

        self.sess = sv.PrepareSession(config=sess_config)

    def train():
        # Initialize progress step
        progress_bar = tqdm(total=config.max_step)
        step = self.sess.run(self.VAE.global_step)
        progress_bar.update(step)
        is_print = (lambda step : step % config.print_step == 0)

        # Main loop
        while not(self.sv.should_stop()):
            progress_bar.update(1)
            step += 1
            if step > config.max_step:
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
                      result["AE_loss"], result["KL_loss"], result['KL_wight']))
                (in_ids, out_ids) = (result['input_ids'], result['sampled_ids'])
                print_ids_to_words(in_ids, out_ids, id_to_word,
                                   sentence_num=5, max_words=40)
                #import pdb; pdb.set_trace()
                #raw_input("Press Enter to continue...")

    def sample():
        while(True):
            z = tf.random_normal([batch_size, self.VAE.hidden_size], name='z')
            sampled_ids = self.sess.run(self.VAE.sample_ids, {self.z : z})
            sampled_ids




    def _ids_to_words(word_ids, id_to_word):
        return [id_to_word[word_id] for word_id in word_ids]
        # if (word_id!=PAD_ID)] # and word_id!=EOS_ID)]

    def _print_ids_to_words(in_ids, out_ids, id_to_word, sentence_num, max_words):
        sentence_list = []
        aster_num = 120
        print("*"*aster_num)
        for i in range(sentence_num):
            # covert from ids to words using 'id_to_word' list
            in_words = _ids_to_words(ids[i], id_to_word)
            out_words = _ids_to_words(ids[i], id_to_word)
            # if sentence is too long, cut it short and punctuate with ellipsis(...)
            if len(in_words) > max_words:
                in_word = in_words[:max_words].append("...")
                out_word = out_words[:max_words].append("...")
            in_str = " ".join(in_words)
            out_str = " ".join(out_words)

            print("[X] " + in_str + "\n" + "[Y] " + out_str)
            print("*"*aster_num)
