# coding=utf-8
# Copyright 2021-Present The THUCTC Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def _lookup(x, vocab, to_cpu=False):
    x = x.tolist()
    y = []

    for _, batch in enumerate(x):
        ids = []
        for _, v in enumerate(batch):
            ids.append(vocab[v] if v in vocab else 2)
        y.append(ids)

    if to_cpu:
        return torch.LongTensor(np.array(y, dtype="int32"))

    return torch.LongTensor(np.array(y, dtype="int32")).cuda()


def load_vocabulary(filename):
    vocab = []
    with open(filename, "rb") as fd:
        for line in fd:
            vocab.append(line.strip())

    word2idx = {}
    idx2word = {}

    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    return vocab, word2idx, idx2word


def lookup(inputs, mode, params, to_cpu=False):
    if mode != "infer":
        features, labels = inputs
        source = features["source"].numpy()
        label = labels['label'].numpy()
        src_mask = torch.FloatTensor(features["source_mask"].numpy())
        label_mask = torch.FloatTensor(labels["label_mask"].numpy())
        correct_mask = torch.FloatTensor(labels["correct_mask"].numpy())
        incorrect_mask = torch.FloatTensor(labels["incorrect_mask"].numpy())

        if not to_cpu:
            src_mask = src_mask.cuda()
            label_mask = label_mask.cuda()
            correct_mask = correct_mask.cuda()
            incorrect_mask = incorrect_mask.cuda()

        source = _lookup(source, params.lookup["source"], to_cpu=to_cpu)
        label = _lookup(label, params.lookup["label"], to_cpu=to_cpu)

        features = {
            "source": source,
            "source_mask": src_mask,
        }
        
        labels = {
            "label": label,
            "label_mask": label_mask,
            "correct_mask": correct_mask,
            "incorrect_mask": incorrect_mask
        }

        return features, labels

    source = inputs["source"].numpy()
    source = _lookup(source, params.lookup["source"], to_cpu=to_cpu)
    src_mask = torch.FloatTensor(inputs["source_mask"].numpy())

    if not to_cpu:
        src_mask = src_mask.cuda()

    features = {
        "source": source,
        "source_mask": src_mask
    }

    return features
