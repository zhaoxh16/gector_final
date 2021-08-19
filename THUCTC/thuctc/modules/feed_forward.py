# coding=utf-8
# Copyright 2021-Present The THUCTC Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import thuctc.utils as utils

from thuctc.modules.module import Module
from thuctc.modules.affine import Affine


def gelu(x):
    # if torch.__version__ >= "1.4.0":
    #     return nn.functional.gelu(x)

    r"""Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    r"""Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *\
        (x + 0.044715 * torch.pow(x, 3.0))))


class FeedForward(Module):

    def __init__(self, input_size, hidden_size, output_size=None,
                 dropout=0.0, act_fn="relu", name="feed_forward"):
        super(FeedForward, self).__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.dropout = dropout

        if act_fn == "gelu":
            self.act_fn = gelu
        elif act_fn == "gelu_new":
            self.act_fn = gelu_new
        else:
            self.act_fn = nn.functional.relu

        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size,
                                           name="output_transform")

        self.reset_parameters()

    def forward(self, x):
        h = self.act_fn(self.input_transform(x))
        h = nn.functional.dropout(h, self.dropout, self.training)
        return self.output_transform(h)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)
