#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-Present The THUCTC Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from collections import OrderedDict


def parseargs():
    parser = argparse.ArgumentParser(
        description="Convert official pre-trained model to ours"
    )

    parser.add_argument("--src_pfile", type=str, required=True,
                        help="official model's parameters name file")
    parser.add_argument("--tgt_pfile", type=str, required=True,
                        help="our model's parameters name file")
    parser.add_argument("--src_model", type=str, required=True,
                        help="official model path")
    parser.add_argument("--tgt_model", type=str, required=True,
                        help="our model path")
    parser.add_argument("--output", type=str, default="model.pt",
                        help="name of output model")
    parser.add_argument("--half", action="store_true",
                        help="convert parameters to float16")

    return parser.parse_args()


def build_mapping(s_pfile, t_pfile):
    fd_s = open(s_pfile, "r", encoding="utf-8")
    fd_t = open(t_pfile, "r", encoding="utf-8")
    n_mapping = {}

    for line in fd_s:
        t_name = fd_t.readline().split(": ")[0]
        s_name = line.split(": ")[0]
        n_mapping[s_name] = t_name

    fd_s.close()
    fd_t.close()

    return n_mapping


def convert_model(name_mapping, s_model, t_model, o_model, half=False):
    src_model = torch.load(s_model, map_location="cpu")
    tgt_model = torch.load(t_model, map_location="cpu")["model"]
    out_model = OrderedDict()
    dtype = torch.float16 if half else torch.float32

    for s_name in src_model:
        if s_name in name_mapping:
            t_name = name_mapping[s_name]
            s_shape = src_model[s_name].shape
            t_shape = tgt_model[t_name].shape

            if s_shape != t_shape:
                raise ValueError(
                    "The shape of {} in the source model is not matched "
                    "with the shape of {} in the target model, {} vs. {}"
                    "".format(s_name, t_name, str(s_shape), str(t_shape))
                )
            else:
                print("{} ==> {}".format(s_name, t_name))

            out_model[t_name] = src_model[s_name].to(dtype)

    for t_name in tgt_model:
        if t_name not in out_model:
            print("Parameter (shape: {}): {} is from the target model: "
                  "{}".format(str(tgt_model[t_name].shape), t_name, t_model))

            out_model[t_name] = tgt_model[t_name].to(dtype)

    torch.save(o_model, {"model": out_model})

    print("Save model: {}".format(o_model))


def main(args):
    n_mapping = build_mapping(args.src_pfile, args.tgt_pfile)
    convert_model(n_mapping, args.src_model,
                  args.tgt_model, args.output, half=args.half)


if __name__ == "__main__":
    main(parseargs())
