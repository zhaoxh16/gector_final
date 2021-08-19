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
from thuctc.modules.layer_norm import LayerNorm


class PositionalEmbedding(torch.nn.Module):

    def __init__(self):
        super(PositionalEmbedding, self).__init__()

    def forward(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("The rank of input must be 3.")

        length = inputs.shape[1]
        channels = inputs.shape[2]
        half_dim = channels // 2

        positions = torch.arange(length, dtype=inputs.dtype,
                                 device=inputs.device)
        dimensions = torch.arange(half_dim, dtype=inputs.dtype,
                                  device=inputs.device)

        scale = math.log(10000.0) / float(half_dim - 1)
        dimensions.mul_(-scale).exp_()

        scaled_time = positions.unsqueeze(1) * dimensions.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)

        if channels % 2 == 1:
            pad = torch.zeros([signal.shape[0], 1], dtype=inputs.dtype,
                              device=inputs.device)
            signal = torch.cat([signal, pad], axis=1)

        return inputs + torch.reshape(signal, [1, -1, channels]).to(inputs)


class Embedding(Module):

    def __init__(self, embed_nums, embed_dims, bias=False, name="embedding"):
        super(Embedding, self).__init__(name=name)

        self.embed_nums = embed_nums
        self.embed_dims = embed_dims

        with utils.scope(name):
            self.weight = nn.Parameter(
                torch.empty(self.embed_nums, self.embed_dims))
            self.add_name(self.weight, "weight")

            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.embed_dims))
                self.add_name(self.bias, "bias")
            else:
                self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0,
                        std=self.embed_dims ** -0.5)

    def forward(self, inputs):
        outputs = nn.functional.embedding(inputs, self.weight)

        if self.bias is not None:
            outputs = outputs + self.bias

        return outputs


class UnifiedEmbedding(Module):

    def __init__(self, params, pos_embed=None, type_embed=False,
                 layer_norm=False, dropout=0.0, scale=False, name="embedding"):
        super(UnifiedEmbedding, self).__init__(name=name)

        self.pos_embed = pos_embed
        self.type_embed = type_embed
        self.vocab_size = len(params.vocabulary["source"])
        self.embedding_size = params.embedding_size
        self.layer_norm = None
        self.out_dropout = None
        self.scale = scale

        if dropout > 0:
            self.out_dropout = nn.Dropout(p=dropout)

        with utils.scope(name):
            self.word_embeddings = Embedding(self.vocab_size,
                                             self.embedding_size,
                                             name="word_embedding")

            if self.pos_embed is not None:
                if self.pos_embed == "learnable":
                    self.pos_embeddings = Embedding(params.max_pos,
                                                    self.embedding_size,
                                                    name="pos_embedding")
                elif self.pos_embed == "functional":
                    self.pos_embeddings = PositionalEmbedding()
                else:
                    raise ValueError("Unsupported position "
                                     "embedding: %s" % pos_embed)

            if self.type_embed:
                self.type_embeddings = Embedding(params.type_vocab_size,
                                                 self.embedding_size,
                                                 name="type_embedding")

            if layer_norm:
                self.layer_norm = LayerNorm(self.embedding_size,
                                            eps=params.layer_norm_eps)

    def resize_word_embedding(self, new_vocab_size): 
        old_embeddings = self.word_embeddings
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_embeddings = Embedding(new_vocab_size,
                                   old_embedding_dim,
                                   name="word_embedding").to(old_embeddings.weight)
        new_embeddings.reset_parameters()
        new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data
        self.word_embeddings = new_embeddings
        self.vocab_size = new_vocab_size

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        inp_shape = input_ids.size()
        inp_length = inp_shape[1]

        inputs = self.word_embeddings(input_ids)

        if self.scale:
            inputs = inputs * (self.embedding_size ** 0.5)

        if self.pos_embed is not None:
            if self.pos_embed == "learnable":
                if position_ids is None:
                    position_ids = torch.arange(inp_length).to(input_ids)
                    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

                inputs = inputs + self.pos_embeddings(position_ids)
            elif self.pos_embed == "functional":
                inputs = self.pos_embeddings(inputs)

        if self.type_embed:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)

            inputs = inputs + self.type_embeddings(token_type_ids)

        if self.layer_norm is not None:
            inputs = self.layer_norm(inputs)

        if self.out_dropout is not None:
            inputs = self.out_dropout(inputs)

        return inputs
