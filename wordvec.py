import os
from tqdm import tqdm
import tensorflow as tf
import numpy as np

DIR = 'data/glove'
FLAGS = tf.app.flags.FLAGS

_glove_spec = {
    '6B' : [50, 100, 200, 300],
    '42B' : [300],
}

def _load_embedding(fname, embed_dim, word_to_id, delimiter=' '):
    glove_vocab = list()
    word2embed = dict()
    num_lines = sum(1 for line in open(fname))
    with open(fname) as lines:
        for line in tqdm(lines, desc=lines.name, total=num_lines):
            line = line.strip().split(delimiter)
            word = line[0]
            embed = line[1:]
            glove_vocab.append(word)
            word2embed[word] = embed

    unknown_words = set(word_to_id.keys()) - set(glove_vocab)
    print("[!] GloVe embeddings have been loaded. {} missing words."\
          .format(len(unknown_words)))
    embed_mat = np.ndarray([len(word_to_id), embed_dim], dtype=np.float32)

    for word, idx in word_to_id.items():
        embed_mat[idx] = word2embed.get(word, np.random.normal(0, 1, embed_dim))

    return embed_mat

def load_glove_embeddings(data_dir, num_tokens, embed_dim, word_to_id):
    # num_tokens : 6B or 42B
    assert(num_tokens in _glove_spec.keys())
    assert(embed_dim in _glove_spec[num_tokens])

    fname = 'glove.{token}.{dim}d.txt'.format(token=num_tokens, dim=embed_dim)
    fpath = os.path.join(FLAGS.glove_dir, fname)
    #glove_vocab = _load_vocab(fpath)
    return _load_embedding(fpath, embed_dim, word_to_id)
