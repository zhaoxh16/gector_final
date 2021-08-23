# coding=utf-8
# Copyright 2021-Present The THUCTC Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import thuctc.utils as utils
import thuctc.modules as modules


class AttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        self.out_dropout = nn.Dropout(params.hidden_dropout)

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(
                params.hidden_size, params.num_heads, params.attention_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size,
                                                eps=params.layer_norm_eps)

    def forward(self, x, bias, memory=None, state=None):
        y = self.attention(x, bias, memory, None)
        y = self.out_dropout(y)
        y = self.layer_norm(x + y)

        return y


class FFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn"):
        super(FFNSubLayer, self).__init__(name=name)

        self.out_dropout = nn.Dropout(params.hidden_dropout)

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(params.hidden_size,
                                                 params.hidden_size * 4,
                                                 act_fn="gelu")
            self.layer_norm = modules.LayerNorm(params.hidden_size,
                                                eps=params.layer_norm_eps)

    def forward(self, x):
        y = self.ffn_layer(x)
        y = self.out_dropout(y)
        y = self.layer_norm(x + y)

        return y


class TransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params)
            self.feed_forward = FFNSubLayer(params)

    def forward(self, x, bias):
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(TransformerEncoder, self).__init__(name=name)
        self.num_encoder_layers = params.num_encoder_layers

        with utils.scope(name):
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])

    def forward(self, x, bias):
        for layer in self.layers:
            x = layer(x, bias)

        return x


class Electra(modules.Module):

    def __init__(self, params, name="electra"):
        super(Electra, self).__init__(name=name)
        self.params = params
        self.keep_idx = self.params.lookup['label'][b'$KEEP']
        self.padding_idx = self.params.lookup['label'][b'@@PADDING@@']
        self.unk_idx = self.params.lookup['label'][b'@@UNKNOWN@@']

        with utils.scope(name):
            self.embedding = modules.UnifiedEmbedding(params,
                                                      pos_embed="learnable",
                                                      type_embed=True,
                                                      layer_norm=True,
                                                      dropout=params.hidden_dropout)
            self.encoder = TransformerEncoder(params)
            self.output_layer = modules.Affine(params.hidden_size,
                                               params.class_num,
                                               name="output_affine")
            self.output_detection_layer = modules.Affine(params.hidden_size, 4, name="output_detection_affine")

        self.criterion = modules.SmoothedCrossEntropyLoss(params.label_smoothing)
        self.hidden_size = params.hidden_size
        self.reset_parameters()

    @property
    def src_embedding(self):
        return self.embedding

    @property
    def softmax_embedding(self):
        return self.softmax_weights

    def freeze_bert(self, freeze:bool):
        self.embedding.requires_grad_(not freeze)
        self.encoder.requires_grad_(not freeze)

    def resize_word_embedding(self, new_vocab_size):
        self.embedding.resize_word_embedding(new_vocab_size)

    def reset_parameters(self):
        self.output_layer.reset_parameters()

    def encode(self, features):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        src_type = features.get("source_type", None)

        inputs = self.embedding(src_seq, token_type_ids=src_type)
        enc_attn_bias = self.masking_bias(src_mask, dtype=inputs.dtype)

        enc_attn_bias = enc_attn_bias.to(inputs.device)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        return encoder_output

    def classify(self, state):
        logits = self.output_layer(state)
        detection_logits = self.output_detection_layer(state)

        return logits, detection_logits

    def convert_to_detection_label(self, label):
        detection_label = torch.ones_like(label)
        detection_label[label == self.padding_idx] = 2
        detection_label[label == self.unk_idx] = 3
        detection_label[label == self.keep_idx] = 0
        return detection_label

    def forward(self, features, labels):
        # labels shape: [batch_size, seq_len]
        # shape: [batch_size, seq_len, hidden_size]
        state = self.encode(features)
        logits, detection_logits = self.classify(state)
        label_seq = labels["label"]
        label_mask = labels["label_mask"]
        loss = self.criterion(logits, label_seq)
        d_label_seq = self.convert_to_detection_label(label_seq)
        d_loss = self.criterion(detection_logits, d_label_seq)
        loss = loss / 2 + d_loss / 2
        mask = label_mask.to(torch.float32)

        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        return (torch.sum(loss * mask) / torch.sum(mask)).to(logits)

    @staticmethod
    def masking_bias(mask, inf=-1e9, dtype=None):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="[PAD]",
            bos="[CLS]",
            eos="[SEP]",
            unk="[UNK]",
            label_unk="@@UNKNOWN@@",
            label_pad="@@PADDING@@",
            class_num=2,
            embedding_size=1024,
            hidden_size=1024,
            num_heads=16,
            num_encoder_layers=24,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            max_pos=512,
            type_vocab_size=2,
            label_smoothing=0.0,
            # Override default parameters
            warmup_steps=0,
            train_steps=200000,
            learning_rate=5e-5,
            learning_rate_schedule="linear_warmup_linear_decay",
            batch_size=32,
            fixed_batch_size=True,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            clip_grad_norm=1.0,
            layer_norm_eps=1e-5
        )

        return params

    @staticmethod
    def default_params(name=None):
        return Electra.base_params()
