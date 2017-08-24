import os

from nltk.tokenize import word_tokenize
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


def load_simple_questions(lower=True):
    """
    Returns:
        (train, valid, vocab)
        train: training set (questions, answers)
        valid: validation set (questions, answers)
        vocab: set(str)
    """
    data_dir = os.path.join(FLAGS.data_dir, 'simple_questions')
    vocab = set()

    def parse_file(lines, vocab):
        questions = []
        answers = []
        for line in lines:
            if lower:
                line = line.rstrip().lower()
            else:
                line = line.rstrip()
            ques, ans = line.split('\t')

            ans_tokens = word_tokenize(ans)
            answers.append(ans_tokens)

            ques = ques[2:]  # remove heading number and space
            ques_tokens = word_tokenize(ques)
            questions.append(ques_tokens)

            vocab.update(ques_tokens + ans_tokens)
            # import ipdb; ipdb.set_trace()
        return questions, answers

    with open(os.path.join(data_dir, 'train.txt')) as lines:
        train = parse_file(lines, vocab)

    with open(os.path.join(data_dir, 'valid.txt')) as lines:
        valid = parse_file(lines, vocab)
    # import ipdb; ipdb.set_trace()
    return train, valid, vocab
