from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import json
from datetime import datetime
from shutil import copyfile
from ctrl_vae_trainer import CtrlVAETrainer

"""
Usage:
    if you want to load pre-trained model,
    set --model_name flag as the path of existing checkpoint file.
    e.g. $python main.py --model_dir=models/my_model_0727_131527
"""
flags = tf.app.flags
flags.DEFINE_string("data_dir", "data", "data directory")
flags.DEFINE_string("glove_dir", "data/glove", "GloVe wordvector directory")
flags.DEFINE_string("config_dir", "config", "config(json file) directory")
flags.DEFINE_string("model_dir", "models", "model super-directory")
flags.DEFINE_string("model_name", None, "model name(sub-directory)")
flags.DEFINE_boolean("is_train", True, "set false when sampling")
FLAGS = tf.app.flags.FLAGS

class ConfigStruct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
        #if not self.__dict__.get("kl_min"):
        #    self.__dict__.update({ "kl_min": None })
    def update(self, **entries):
        self.__dict__.update(entries)

def main(_):

    # Model sub-directory = model_dir/model_name_{datatime}
    time_str = datetime.now().strftime('%m%d_%H%M%S')
    if FLAGS.model_name:
        FLAGS.model_subdir = os.path.join(FLAGS.model_dir, FLAGS.model_name)
    else:
        FLAGS.model_name = "new_model"
        FLAGS.model_name = "{}_{}".format(FLAGS.model_name, time_str)
        FLAGS.model_subdir = os.path.join(FLAGS.model_dir, FLAGS.model_name)
    print("[*] MODEL directory: %s" % FLAGS.model_subdir)

    # Make directories if not exists
    for path in [FLAGS.data_dir, FLAGS.model_subdir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Load config.json file & copy it as log file in the model's sub-dir
    config_path = os.path.join(FLAGS.config_dir, "config.json")
    config_log_path = os.path.join(FLAGS.model_subdir, "config.json")
    copyfile(config_path, config_log_path)
    with open(config_path) as config_file:
        configs = json.load(config_file)

    # Generate ConfigStruct instance
    config = ConfigStruct(**configs["model"])
    config.update(**configs["train"])
    config.update(**configs["sampling"])
    config.update(**configs["data"])

    # Trainer
    trainer = CtrlVAETrainer(config)

    if FLAGS.is_train:
        trainer.train()
    else:
        if config.interpolate_samples:
            trainer.interpolate_samples()
        else:
            trainer.sample()

if __name__ == "__main__":
    tf.app.run()
