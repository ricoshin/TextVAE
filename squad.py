import os
import json

from nltk.tokenize import word_tokenize
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

def _get_tokenizer():
    def tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"')
                for token in word_tokenize(tokens)]
    return tokenize

def _read_para(para, tokenize, pasg_list, ques_list, answ_list,
               answ_index_list, word_vocab):
    pasg = para['context']
    pasg = pasg.replace("''", '" ').replace("``", '" ').lower()
    pasg_tokens = tokenize(pasg)

    word_vocab.update(pasg_tokens)

    for qa in para['qas']:
        ques = qa['question'].lower()
        ques_tokens = tokenize(ques)

        word_vocab.update(ques_tokens)

        pasg_list.append(pasg_tokens)
        ques_list.append(ques_tokens)

        answ = qa['answers'][:1][0]['text'].lower()
        answ_tokens = tokenize(answ)

        word_vocab.update(answ_tokens)

        answ_list.append(answ_tokens)

        answ_pairs = _get_word_idxs(pasg, pasg_tokens, qa['answers'])
        answ_index_list.append(answ_pairs)

    # import ipdb; ipdb.set_trace()
    # print("test")

def _get_word_idxs(pasg, pasg_tokens, answers):
    answ_pairs = []
    for ans in answers[:1]:
        char_p1 = ans['answer_start']
        char_p2 = char_p1 + len(ans['text']) - 1
        char_idx = 0
        for word_idx, token in enumerate(pasg_tokens):
            found = pasg.find(token, char_idx)
            if found < 0:
                raise Exception('Could not find {token} in the following passage'
                                'with the start index {start}:\n{pasg}'
                                ''.format(token=token, start=char_idx, pasg=pasg))
            char_idx = found + len(token)
            if found <= char_p1 < char_idx:
                word_p1 = word_idx
            if found <= char_p2 < char_idx:
                word_p2 = word_idx
                # answ_pairs.append([word_p1, word_p2])
                answ_pairs = [word_p1, word_p2]
                break
    assert(answ_pairs)
    return answ_pairs

def load_squad(lower=True):
    """
    Returns:
        (train, valid, vocab)
        train: training set (questions, answers)
        valid: validation set (questions, answers)
        vocab: set(str)
    """
    data_dir = os.path.join(FLAGS.data_dir, 'squad')
    vocab = set()

    def parse_file(lines, vocab):
        tokenize = _get_tokenizer()
        pasg_list = list()
        ques_list = list()
        answ_list = list()
        answ_index_list = list()

        json_dict = json.load(lines)
        data = json_dict['data']
        for article in data:
            for para in article['paragraphs']:
                _read_para(para, tokenize, pasg_list, ques_list, answ_list,
                           answ_index_list, vocab)

        return ques_list, answ_list, answ_index_list, pasg_list

    with open(os.path.join(data_dir, 'train/dev-v1.1.json')) as lines:
        train = parse_file(lines, vocab)
        # import ipdb; ipdb.set_trace()
    # with open(os.path.join(data_dir, 'valid/dev-v1.1.json')) as lines:
    #     valid = parse_file(lines, vocab)

    return train, vocab #,valid
