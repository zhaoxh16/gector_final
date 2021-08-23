# coding=utf-8
# Copyright 2021-Present The THUCTC Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import tensorflow as tf


def sort_input_file(filename, reverse=True):
    with open(filename, "rb") as fd:
        inputs = [line.strip() for line in fd]
    
    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]
    
    sorted_input_lens = sorted(input_lens,
                               key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []
    
    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i
    
    return sorted_keys, sorted_inputs


def build_input_fn(filenames, mode, params):
    def train_input_fn():
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())
        dataset = dataset.prefetch(params.buffer_size)
        dataset = dataset.shuffle(params.buffer_size)

        # Split string
        dataset = dataset.map(
            lambda x, y: (tf.strings.split([x]).values,
                          tf.strings.split([y]).values),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Append BOS and EOS
        dataset = dataset.map(
            lambda x, y:(
                tf.concat(
                    [[tf.constant(params.bos)], x[1:], [tf.constant(params.eos)]],
                    axis=0
                ),
                tf.concat(
                    [y, [tf.constant(params.label_pad)]],
                    axis=0
                )
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.map(
            lambda x, y: (
                {"source": x, "source_length": tf.shape(x)[0]}, 
                {"label": y, "label_length": tf.shape(y)[0]},
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        def element_length_func(x, y):
            return x["source_length"]

        def valid_size(x, y):
            size = element_length_func(x, y)
            return tf.logical_and(size >= min_length, size <= max_length)

        # transformation_fn = tf.data.experimental.bucket_by_sequence_length(
        #     element_length_func,
        #     boundaries,
        #     batch_sizes,
        #     padded_shapes=({
        #             "source": tf.TensorShape([None]),
        #             "source_length": tf.TensorShape([])
        #         }, {
        #             "label": tf.TensorShape([None]),
        #             "label_length": tf.TensorShape([])
        #         }),
        #     padding_values=({
        #             "source": params.pad,
        #             "source_length": 0
        #         }, {
        #             "label": params.label_pad,
        #             "label_length": 0
        #         }),
        #     pad_to_bucket_boundary=False
        # )

        dataset = dataset.filter(valid_size)
        # dataset = dataset.apply(transformation_fn)
        dataset = dataset.padded_batch(
            params.batch_size,
            padded_shapes=(
                {
                    "source": tf.TensorShape([None]),
                    "source_length": tf.TensorShape([])
                },
                {
                    "label": tf.TensorShape([None]),
                    "label_length": tf.TensorShape([])
                }),
            padding_values=(
                {
                    "source": params.pad,
                    "source_length": 0
                },
                {
                    "label": params.label_pad,
                    "label_length": 0
                }),
        )

        dataset = dataset.map(
            lambda x, y: (
                {
                    "source": x["source"],
                    "source_mask": tf.sequence_mask(x["source_length"],
                                                    tf.shape(x["source"])[1],
                                                    tf.float32)
                },
                {
                    "label": y["label"],
                    "label_mask": tf.sequence_mask(y["label_length"] - 1,
                                                   tf.shape(y["label"])[1],
                                                   tf.float32)
                }
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return dataset

    # def eval_input_fn():
    #     src_dataset = tf.data.TextLineDataset(filenames[0])
    #     tgt_dataset = tf.data.TextLineDataset(filenames[1])
    #     dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    #     dataset = dataset.shard(torch.distributed.get_world_size(),
    #                             torch.distributed.get_rank())
    #     dataset = dataset.prefetch(params.buffer_size)

    #     # Split string
    #     dataset = dataset.map(
    #         lambda x, y: (tf.strings.split([x]).values,
    #                       tf.strings.split([y]).values),
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #     # Append BOS and EOS
    #     dataset = dataset.map(
    #         lambda x, y:(
    #             tf.concat(
    #                 [[tf.constant(params.bos)], x, [tf.constant(params.eos)]],
    #                 axis=0
    #             ),
    #             tf.concat([y, [tf.constant(params.label_pad)]], axis=0)),
    #             num_parallel_calls=tf.data.experimental.AUTOTUNE
    #         )

    #     dataset = dataset.map(
    #         lambda x, y: (
    #             {"source": x, "source_length": tf.shape(x)[0]}, 
    #             {"label": y, "label_length": tf.shape(y)[0]},
    #         ),
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     )

    #     # Batching
    #     dataset = dataset.padded_batch(
    #         params.decode_batch_size,
    #         padded_shapes=(
    #             {
    #                 "source": tf.TensorShape([None]),
    #                 "source_length": tf.TensorShape([])
    #             },
    #             {
    #                 "label": tf.TensorShape([None]),
    #                 "label_length": tf.TensorShape([])
    #             }
    #         ),
    #         padding_values=(
    #             {
    #                 "source": params.pad,
    #                 "source_length": 0
    #             },
    #             {
    #                 "label": params.label_pad,
    #                 "label_length": 0
    #             }
    #         )
    #     )

    #     dataset = dataset.map(
    #         lambda x, y: (
    #             {
    #                 "source": x["source"],
    #                 "source_mask": tf.sequence_mask(x["source_length"],
    #                                                 tf.shape(x["source"])[1],
    #                                                 tf.float32)
    #             },
    #             {
    #                 "label": y["label"],
    #                 "label_mask": tf.sequence_mask(y["label_length"] - 1,
    #                                                tf.shape(y["label"])[1],
    #                                                tf.float32)
    #             }
    #         ),
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     )

    #     return dataset

    def infer_input_fn():
        sorted_key, sorted_data = sort_input_file(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(sorted_data))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.map(
            lambda x: tf.strings.split([x]).values,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x: tf.concat([[tf.constant(params.bos)], x, [tf.constant(params.eos)]], axis=0),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x: {
                "source": x,
                "source_length": tf.shape(x)[0]
            },
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.padded_batch(
            params.decode_batch_size,
            padded_shapes={
                "source": tf.TensorShape([None]),
                "source_length": tf.TensorShape([])
            },
            padding_values={
                "source": tf.constant(params.pad),
                "source_length": 0
            })

        dataset = dataset.map(
            lambda x: {
                "source": x["source"],
                "source_mask": tf.sequence_mask(x["source_length"],
                                                tf.shape(x["source"])[1],
                                                tf.float32),
            },
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return sorted_key, dataset

    if mode == "train":
        return train_input_fn
    # if mode == "eval":
    #     return eval_input_fn
    elif mode == "infer":
        return infer_input_fn
    else:
        raise ValueError("Unknown mode {}".format(mode))


def get_dataset(filenames, mode, params):
    input_fn = build_input_fn(filenames, mode, params)

    with tf.device("/cpu:0"):
        dataset = input_fn()

    return dataset
