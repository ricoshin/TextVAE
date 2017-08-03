import numpy as np
import tensorflow as tf

from data import load_simple_questions_dataset
from disc_model import Discriminator


class DTrainer(object):
    def __init__(self, config, train, valid, word_embd, word2idx):
        self.train_data = train
        self.valid_data = valid
        self.word2idx = word2idx
        self.word_embd = word_embd
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.num_examples = train[0].shape[0]
        self.max_ques_len = train[0].shape[1]
        self.config = config
        self.model = Discriminator(cfg=config,
                                   word_embd=word_embd,
                                   max_ques_len=self.max_ques_len)

    def random_batch_generator(self, data, batch_size, num_steps):
        for i in range(num_steps):
            idx = np.random.permutation(self.num_examples)[:batch_size]
            batch_ques = data[0][idx]
            batch_ans = data[1][idx].reshape(-1)
            yield batch_ques, batch_ans

    def _get_ques_len(self, questions):
        ques_len = []
        for ques in questions:
            for n, wi in enumerate(ques):
                if self.idx2word[wi] == '-pad-':
                    break
            ques_len.append(n)
        return ques_len

    def train(self):
        self.optimizer = tf.train.AdamOptimizer
        opt = tf.train.AdamOptimizer(1)
        train_op = opt.minimize(self.model.loss)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        sv = tf.train.Supervisor(logdir='logs',
                                 save_model_secs=60)
        with sv.managed_session(config=sess_config) as sess:
            for i, batch in enumerate(self.random_batch_generator(
                                          self.train_data,
                                          self.config.batch_size,
                                          self.config.max_step)):
                if sv.should_stop():
                    break
                questions, answers = batch
                fetch_dict = dict(
                    pred=self.model.pred,
                    loss=self.model.loss,
                    opt=train_op,
                )
                ques_len = self._get_ques_len(questions)
                feed_dict = {
                    self.model.ques: questions,
                    self.model.answ: answers,
                    self.model.ques_len: ques_len,
                }
                result = sess.run(fetch_dict, feed_dict)
                if i % 500 == 0:
                    print('Loss:', result['loss'])
                    self._print_log(questions, ques_len, answers, result['pred'])

    def _print_log(self, ques, ques_len, answ, pred):
        print('%60s |%15s | %s' % ('Question', 'Prediction', 'Answer'))
        for i, (q, l, a, p) in enumerate(zip(ques, ques_len, answ, pred)):
            question = ' '.join(map(self.idx2word.get, q[:l]))
            answer = self.idx2word[a]
            prediction = self.idx2word[p]
            print('%60s |%15s | %s' % (question, prediction, answer))
            if i == 5:
                break



def main():
    class Config(object):
        pass
    config = Config()
    config.data_dir = 'data'
    config.embed_dim = 50
    config.hidden_size = 64
    config.batch_size = 100
    config.max_step = 100000
    train, valid, word_embd, word2idx = load_simple_questions_dataset(config)
    trainer = DTrainer(config, train, valid, word_embd, word2idx)
    trainer.train()

if __name__ == '__main__':
    main()

