# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import gzip
import json
import os
from nltk.tokenize import word_tokenize
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

Tokens = namedtuple('Tokens', ['words', 'charss'])
SquadFile = namedtuple('SquadFile', ['pasg', 'ques', 'answ',
                                     'wd_voc', 'ch_voc'])


def load_marco(cfg):
    data_dir = os.path.join(FLAGS.data_dir, 'marco')
    vocab = set()
    # tokenize = _get_tokenizer(cfg)
    # pasg_list = list()
    # pasg_selected_list = list()
    # ques_list = list()
    # answ_list = list()
    # word_vocab = set()
    # char_vocab = set()
    # data = json_dict['data'] # SQuAD는 data에서부터 시작
    # # import pdb; pdb.set_trace()
    # for idx,para in enumerate(data):
    #     if idx < 10:
    #         _read_para(cfg, para, tokenize,
    #                pasg_list, ques_list, answ_list,
    #                word_vocab, char_vocab)
    # return SquadFile(pasg=pasg_list, ques=ques_list, answ=np.array(answ_list),
    #                  wd_voc=word_vocab, ch_voc=char_vocab)
    def parse_file(lines, vocab):
        tokenize = _get_tokenizer()
        pasg_list = list()
        ques_list = list()
        answ_list = list()

        data = lines['data']

        for para in data:
            _read_para(para, tokenize, pasg_list, ques_list, answ_list, vocab)

        return ques_list, answ_list, pasg_list
        # for para in data:
        #     for para in article['paragraphs']:
        #         _read_para(para, tokenize, pasg_list, ques_list, answ_list,
        #                    answ_index_list, vocab)

        # return ques_list, answ_list, answ_index_list, pasg_list

    with gzip.open(os.path.join(data_dir, 'dev_v1.1.json.gz')) as f:
        lines = {'data':[]}
        for line in f.readlines():
            lines['data'].append(json.loads(line))
        train = parse_file(lines, vocab)
    # import ipdb; ipdb.set_trace()
    return train, vocab #,valid


def _read_para(para, tokenize, pasg_list, ques_list, answ_list, word_vocab):

    if len(para['answers']) == 0:
        return #index 15 is empty

    pasg = str()

    for pa in para['passages']:
        pasg += pa['passage_text']

    pasg = pasg.replace("''", '" ').replace("``", '" ').lower()
    pasg_tokens = tokenize(pasg)

    ques = para['query'].lower()
    ques_tokens = tokenize(ques)

    answ = para['answers'][0].lower()
    answ_tokens = tokenize(answ)
    # for ans in para['answers']:
    #     answ.append(ans)

    word_vocab.update(pasg_tokens)
    word_vocab.update(ques_tokens)
    word_vocab.update(answ_tokens)

    pasg_list.append(pasg_tokens)
    ques_list.append(ques_tokens)
    answ_list.append(answ_tokens)


#안씀 squad에서만씀
# def _get_word_idxs(pasg, pasg_tokens, answers):
#     answ_pairs = []
#     for ans in answers[:1]:
#         char_p1 = ans['answer_start']
#         char_p2 = char_p1 + len(ans['text']) - 1
#         char_idx = 0
#         for word_idx, token in enumerate(pasg_tokens):
#             found = pasg.find(token, char_idx)
#             if found < 0:
#                 raise Exception('Could not find {token} in the following passage'
#                                 'with the start index {start}:\n{pasg}'
#                                 ''.format(token=token, start=char_idx, pasg=pasg))
#             char_idx = found + len(token)
#             if found <= char_p1 < char_idx:
#                 word_p1 = word_idx
#             if found <= char_p2 < char_idx:
#                 word_p2 = word_idx
#                 answ_pairs.append([word_p1, word_p2])
#                 break
#     assert(answ_pairs)
#     return answ_pairs

def _get_tokenizer():
    def tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"')
                for token in word_tokenize(tokens)]
    return tokenize
