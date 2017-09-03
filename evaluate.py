""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import math
import operator
from functools import reduce
import copy

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def ngram(n, iterable):
    iterable = iter(iterable) 
    window = [next(iterable) for _ in range(n)]
    while True:
        yield window
        window.append(next(iterable))
        window = window[1:]

def bleu_ngram(n, candidate, references):
    pred = [' '.join(window) for window in ngram(n, candidate)]
    truths = [[' '.join(window) for window in ngram(n, reference)] for reference in references]

    ref_counts = Counter()
    for truth in truths:
        ref_counts |= Counter(truth)

    common = Counter(pred) & ref_counts
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    return 1.0 * num_same / len(pred)
    
def bleu_score(prediction, ground_truths, num_ngrams):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truths_tokens = [normalize_answer(ground_truth).split() for ground_truth in ground_truths]

    # brevity penalty
    num_pred = len(prediction_tokens)
    num_truth = min(len(truth) for truth in ground_truths_tokens)
    if 1 <= num_pred <= num_truth:
        penalty = math.exp(1 - 1.0 * num_truth / num_pred) 
    else:
        penalty = 1

    score = 0
    for i in range(1, num_ngrams + 1):
        precision = bleu_ngram(i, prediction_tokens, ground_truths_tokens)
        if precision > 0:
            score += math.log(precision)

    # applying geometric mean
    bleu = math.exp(score / num_ngrams) 
    return bleu * penalty

def chunk(a, b):
    b = copy.deepcopy(b)
    c, u = 0, 0 # c: number of chunks, u: number of words associated with chunk

    # Find a common sequence (= a chunk)
    def _calc_common_length(x, y):
        n = 0
        for cx, cy in zip(x, y):
            if cx != cy:
                break
            n += 1
        return n
    
    def _find(corpus, x, start=0):
        for i, word in enumerate(corpus[start:]):
            if word == x:
                return start + i
        return -1

    for i in range(len(a)):
        max_len = 0
        pos = -1
        j = -1

        # Find a common longest sequence 
        while True:
            j = _find(b, a[i], j + 1)
            if j < 0:
                break
            common_len = _calc_common_length(a[i:], b[j:])
            if common_len > max_len:
                pos = j
                max_len = common_len

        # replace empty sentence ([0])
        if pos >= 0:
            b[pos:pos+max_len] = [0]
            c += 1
            u += max_len
    return c, u
        

def meteor_score(prediction, ground_truth):
    # According to the paper of METEOR, stemming process is required.
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    fmean = 10.0 * precision * recall / (recall + 9 * precision)
    c, u = chunk(prediction_tokens, ground_truth_tokens)
    frag = 1.0 * c / u
    penalty = 0.5 * (frag ** 3)
    return fmean * (1 - penalty)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def simple_evaluate(references, predictions):
    metrics = {
        'em': lambda p, g: metric_max_over_ground_truths(exact_match_score, p, g),
        'f1': lambda p, g: metric_max_over_ground_truths(f1_score, p, g),
        'bleu': bleu_score,
        'meteor': lambda p, g: metric_max_over_ground_truths(meteor_score, p, g),
    }

    total = 0
    scores = { k: 0 for k in metrics }
    for ref, pred in zip(references, predictions):
        total += 1
        for k, f in metrics:
            scores[k] += f(pred, [ref])
    for k in metrics:
        scores[k] = 100.0 * scores[k] / total
    return scores

def evaluate(dataset, predictions):
    f1 = exact_match = bleu = meteor = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                bleu += metric_max_over_ground_truths(
                    bleu_score, prediction, ground_truths)
                meteor += metric_max_over_ground_truths(
                    meteor_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    bleu = 100.0 * bleu / total
    meteor = 100.0 * meteor / total

    return {'exact_match': exact_match, 'f1': f1, 'bleu': bleu, 'meteor': meteor }


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))


