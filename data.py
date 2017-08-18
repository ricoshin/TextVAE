import copy
import csv
import numpy as np
import os
import re
import sys

import tensorflow as tf

from simple_questions import load_simple_questions
from wordvec_sq import load_glove_vocab, load_glove_embeddings

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

EOS_TOKEN = u"<eos>"
DROP_TOKEN = u"<drop>"
UNK_TOKEN = u"<unk>"
PAD_TOKEN = u"<pad>"

PAD_ID = 0
EOS_ID = 1
DROP_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, DROP_TOKEN, UNK_TOKEN]

FLAGS = tf.app.flags.FLAGS


def find_abbr_candidates(text):
    abbr_regex = re.compile('(\s\w+\s([A-Z][a-z]*\.([A-Za-z]\.)*)\s\w+)')
    known_abbrs = ['mr.', 'mrs.', 'dr.', 'st.']
    candidates = [x for x in abbr_regex.findall(text)
                  if x[1].lower() not in known_abbrs]
    cand_dict = dict()
    for cand in candidates:
        examples = cand_dict.get(cand[1].lower())
        if examples is None:
            cand_dict[cand[1].lower()] = []
            examples = cand_dict[cand[1].lower()]
        examples.append(cand[0])
    return cand_dict


def replace_unknowns(sents, unknowns):
    """
    Args:
        sents: list(list(str))
        unknowns: set(str)
    Returns:
        list(list(str))
    """
    def replace(sent):
        return [token if token not in unknowns else UNK_TOKEN for token in sent]
    return list(map(replace, sents))


def append_pads(sents, max_len):
    """
    Args:
        sents: list(list(str))
    Returns:
        list(list(str))
    """
    sents = copy.deepcopy(sents)
    for sent in sents:
        num_pads = max_len-len(sent)
        assert(num_pads >= 0)
        sent.extend([PAD_TOKEN for _ in range(num_pads)])
    return sents


def append_eos(sents):
    sent_lens = []
    for sent in sents:
        sent.append(EOS_TOKEN)
        sent_lens.append(len(sent))
    return sents, sent_lens


def make_reverse(sents):
    sents_rev = []
    for sent in sents:
        sents_rev.append(list(reversed(copy.deepcopy(sent))))
    return sents_rev


def convert_to_idx(sents, word2idx):
    """
    Args:
        sents: list(list(str))
        word2idx: dict(str: number)
    Returns:
        list(list(number))
    """
    return [[word2idx[token] for token in sent] for sent in sents]


def convert_to_token(sents, word2idx, trim=True):
    idx2word = {v: k for k, v in word2idx.items()}
    # TODO: it is not trimming but removing only PAD.
    return [[idx2word[idx] for idx in sent
             if not (trim and idx2word[idx] == PAD_TOKEN)] for sent in sents]


def remove_unknown_answers(data, vocab):
    data = zip(data[0], data[1])

    questions = []
    answers = []
    new_vocab = set()
    for q, a in data:
        if len(a) == 1 and a[0] in vocab:
            questions.append(q)
            answers.append(a)
        elif len(a) > 1 and '_'.join(a) in vocab:
            questions.append(q)
            a = '_'.join(a)
            answers.append([a])
            new_vocab.add(a)
    return (questions, answers), new_vocab


def load_simple_questions_dataset(config):

    data_npz = os.path.join(FLAGS.data_dir, 'data.npz')
    word2idx_txt = os.path.join(FLAGS.data_dir, 'word2idx.txt')

    if (os.path.exists(data_npz) and os.path.exists(word2idx_txt) and
            not config.is_reload_embed):
        npz = np.load(data_npz)
        embd_mat = npz['embd_mat']
        train_ques = npz['train_ques'].astype(np.int32)
        train_ques_rev = npz['train_ques_rev'].astype(np.int32)
        train_ques_len = npz['train_ques_len'].astype(np.int32)
        train_ans = npz['train_ans'].astype(np.int32)
        valid_ques = npz['valid_ques'].astype(np.int32)
        valid_ques_rev = npz['valid_ques_rev'].astype(np.int32)
        valid_ques_len = npz['valid_ques_len'].astype(np.int32)
        valid_ans = npz['valid_ans'].astype(np.int32)

        with open(word2idx_txt) as f:
            reader = csv.reader(f, delimiter='\t')
            word2idx = {row[0]: int(row[1]) for row in reader}

        train = train_ques, train_ques_rev, train_ques_len, train_ans
        valid = valid_ques, valid_ques_rev, valid_ques_len, valid_ans
        return train, valid, embd_mat, word2idx

    glove_vocab = load_glove_vocab(os.path.join(FLAGS.data_dir, 'glove'),
                                   '6B', config.embed_dim)

    train, valid, dataset_vocab = load_simple_questions(config)

    train, new_vocab = remove_unknown_answers(train, glove_vocab)
    dataset_vocab.update(new_vocab)

    valid, new_vocab = remove_unknown_answers(valid, glove_vocab)
    dataset_vocab.update(new_vocab)

    train_q, train_a = train[0], train[1]
    valid_q, valid_a = valid[0], valid[1]

    unknowns = dataset_vocab-glove_vocab
    train_q = replace_unknowns(train_q, unknowns)
    train_a = replace_unknowns(train_a, unknowns)
    valid_q = replace_unknowns(valid_q, unknowns)
    valid_a = replace_unknowns(valid_a, unknowns)
    vocab = dataset_vocab-unknowns

    train_q, train_q_len = append_eos(train_q)
    valid_q, valid_q_len = append_eos(valid_q)

    train_q_rev = make_reverse(train_q)
    valid_q_rev = make_reverse(valid_q)

    max_len = max(len(sent) for sent in train_q+valid_q)
    train_q = append_pads(train_q, max_len)
    train_q_rev = append_pads(train_q_rev, max_len)
    valid_q = append_pads(valid_q, max_len)
    valid_q_rev = append_pads(valid_q_rev, max_len)

    vocab = SPECIAL_TOKENS + list(vocab)
    embd_mat, word2idx = load_glove_embeddings(os.path.join(FLAGS.data_dir,
                                                            'glove'),
                                               '6B', config.embed_dim, vocab)

    train_q = convert_to_idx(train_q, word2idx)
    train_q_rev = convert_to_idx(train_q_rev, word2idx)
    train_a = convert_to_idx(train_a, word2idx)
    valid_q = convert_to_idx(valid_q, word2idx)
    valid_q_rev = convert_to_idx(valid_q_rev, word2idx)
    valid_a = convert_to_idx(valid_a, word2idx)

    with open(word2idx_txt, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(word2idx.items())
    data_dict = dict(embd_mat=embd_mat,
                     train_ques=train_q,
                     train_ques_len=train_q_len,
                     train_ques_rev=train_q_rev,
                     train_ans=train_a,
                     valid_ques=valid_q,
                     valid_ques_len=valid_q_len,
                     valid_ques_rev=valid_q_rev,
                     valid_ans=valid_a)
    np.savez(data_npz, **data_dict)

    train = tuple(map(np.array, [train_q, train_q_rev, train_q_len, train_a]))
    valid = tuple(map(np.array, [valid_q, valid_q_rev, valid_q_len, valid_a]))
    return train, valid, embd_mat, word2idx
