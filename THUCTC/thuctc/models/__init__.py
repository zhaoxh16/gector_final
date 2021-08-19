# coding=utf-8
# Copyright 2021-Present The THUCTC Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thuctc.models.transformer
import thuctc.models.bert
import thuctc.models.electra


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thuctc.models.transformer.Transformer

    if name == "bert":
        return thuctc.models.bert.Bert

    if name == "electra":
        return thuctc.models.electra.Electra

    raise LookupError("Unknown model {}".format(name))
