""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
from functools import reduce
from math import exp
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

def bleu_ngram(n, prediction, ground_truth):
    pred = [' '.join(window) for window in ngram(n, prediction)]
    truth = [' '.join(window) for window in ngram(n, ground_truth)]

    common = Counter(pred) & Counter(truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    return 1.0 * num_same / len(pred)
    
def bleu_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    penalty = 1
    if 1 <= len(prediction_tokens) <= len(ground_truth_tokens):
        penalty = exp(1 - 1.0 * len(ground_truth_tokens) / len(prediction_tokens)) 

    scores = []
    # brevity penalty
    for i in range(1, 5):
        score = scores.append(bleu_ngram(i, prediction, ground_truth))

    # applying geometric mean
    bleu = reduce(lambda x, y: x * y, scores) ** (1.0 / len(scores))
    return bleu * penalty

def chunk(a, b):
    b = copy.deepcopy(b)
    c, u = 0, 0
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

        while True:
            j = _find(b, a[i], j + 1)
            if j < 0:
                break
            common_len = _calc_common_length(a[i:], b[j:])
            if common_len > max_len:
                pos = j
                max_len = common_len
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
        'em': exact_match_score,
        'f1': f1_score,
        'bleu': bleu_score,
        'meteor': meteor_score,
    }

    total = 0
    scores = { k: 0 for k in metrics }
    for ref, pred in zip(references, predictions):
        total += 1
        for k in metrics:
            scores[k] += metric_max_over_ground_truths(
                metrics[k], pred, [ref])
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


