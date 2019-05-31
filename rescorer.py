#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import subprocess
import re


class Hypothesis:

    def __init__(self, hyp, alpha):
        self.translation = ' '.join(hyp.split()[:-1])
        self.empty = True if self.translation == '' else False
        self.nmt_score = float(re.findall(r"[-+]?\d*\.\d+|\d+", hyp)[-1])
        self.fa_score = 0
        self.alpha = alpha

    def is_empty(self):
        return self.empty

    def add_score(self, score):
        self.fa_score = score

    def get_score(self):
        return self.alpha * self.nmt_score + (1 - self.alpha) * self.fa_score

    def get_translation(self):
        return self.translation


class Sentence:

    def __init__(self, source, n_best):
        self.source = source
        self.n_best = n_best
        self.hyps = []

    def add_hyp(self, hyp):
        if not hyp.is_empty():
            self.hyps.append(hyp)

    def get_translation(self, n):
        return self.hyps[n].get_translation()

    def get_hyps(self):
        return self.hyps

    def get_source(self):
        return self.source

    def rescore(self):
        self.hyps.sort(reverse=True, key=lambda hyp: hyp.get_score())


def rescore(sentences, fast_align, t, m, fwd_params):
    text = []

    for sentence in sentences:
        for hyp in sentence.get_hyps():
            text.append(sentence.get_source() + ' ||| ' + hyp.get_translation())

    alignments = subprocess.run((fast_align + ' -i - -d -T ' + str(t) + ' -m ' + str(m)
                          + ' -f ' + fwd_params).split(), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, input='\n'.join(text), encoding='utf-8').stdout.split('\n')

    n = 0
    for sentence in sentences:
        for hyp in sentence.get_hyps():
            hyp.add_score(float(alignments[n].split()[-1]))
            n += 1
        sentence.rescore()


def show_best_hypothesis(sentences):
    for sentence in sentences:
        print(sentence.get_translation(0))


def main(opt):
    """
    Given an n-best list and its NMT score, this scripts re-scores the hypothesis using fast align, and returns
    the best hypothesis for each source sentence.
    :param opt:
    :return:
    """
    sources = open(opt.sources[0], 'r')
    hypothesis = open(opt.translations[0], 'r')
    sentences = []

    # Read source and n-best and build the data structures.
    src = sources.readline().strip()
    while src != '':
        sentence = Sentence(src, opt.n_best[0])
        for n in range(opt.n_best[0]):
            hyp = hypothesis.readline().strip()
            sentence.add_hyp(
                Hypothesis(hyp, opt.alpha[0]))
        sentences.append(sentence)
        src = sources.readline().strip()

    # Align n-best with fast align and re-score them.
    rescore(sentences, opt.fast_align[0], opt.tension[0], opt.target_length[0], opt.forward_params[0])

    # Print best hypothesis for each source sentence.
    show_best_hypothesis(sentences)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Re-scores n-best list using fast align.""", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-s', '--sources', type=str, nargs='+', help='Source file.', required=True)
    parser.add_argument('-t', '--translations', type=str, nargs='+', help='File containing the n-best '
                                                                          'list with their scores.', required=True)
    parser.add_argument('-n', '--n_best', type=int, nargs='+', help='Number of n-best.', required=True)
    parser.add_argument('-b', '--fast_align', type=str, nargs='+', help='Path to fast_align.', required=True)
    parser.add_argument('-T', '--tension', type=float, nargs='+', help='Tension value computed with force_align.py.',
                        required=True)
    parser.add_argument('-m', '--target_length', type=float, nargs='+', help='Target length value computed with'
                                                                             'force_align.py.', required=True)
    parser.add_argument('-f', '--forward_params', type=str, nargs='+', help='fwd_params computed with force_align.py.',
                        required=True)
    parser.add_argument('-a', '--alpha', type=float, nargs='+', help='Linear combination factor.', required=False,
                        default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
