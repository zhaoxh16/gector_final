# coding=utf-8
# Copyright 2021-Present The THUCTC Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import glob
import logging
import operator
import os
import shutil
import time
import torch

import torch.distributed as dist

from thuctc.data.vocab import lookup
from thuctc.utils.checkpoint import save, latest_checkpoint
from thuctc.utils.inference import argmax_encoding, beam_search
from thuctc.utils.bleu import bleu
from thuctc.utils.bpe import BPE
from thuctc.utils.misc import get_global_step
from thuctc.utils.summary import scalar


def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _read_score_record(filename):
    # "checkpoint_name": score
    records = []

    if not os.path.exists(filename):
        return records

    with open(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1].rstrip(".pt"))
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with open(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    # Sort
    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records


def _convert_to_string(tensor, params):
    ids = tensor.tolist()
    output = []

    for wid in ids:
        output.append(params.mapping["label"][wid])

    return output


def f_score(predict, ground_truth):
    TP = 0.0
    d_TP = 0.0
    TP_FP = 0.0
    TP_FN = 0.0

    assert len(predict) == len(ground_truth)
    for i in range(len(predict)):
        assert len(predict[i]) == len(ground_truth[i])
        for pos in range(len(predict[i])):
            predict_label = predict[i][pos]
            gt_label = ground_truth[i][pos]
            if gt_label == b"@@PADDING@@":
                continue

            if gt_label == predict_label and gt_label != b"$KEEP":
                TP += 1.0

            if gt_label != b"$KEEP" and predict_label != b"$KEEP":
                d_TP += 1.0

            if predict_label != b"$KEEP":
                TP_FP += 1.0

            if gt_label != b"$KEEP":
                TP_FN += 1.0

    logging.info("TP: {:.4f}, TP_FP: {:.4f}, " \
        "TP_FN: {:.4f}".format(TP, TP_FP, TP_FN))

    c_p = (TP / TP_FP) if TP_FP > 0 else 0.0
    c_r = (TP / TP_FN) if TP_FN > 0 else 0.0
    c_f1 = (2 * c_p * c_r / (c_p + c_r)) \
        if (c_p + c_r > 0.0) else 0.0
    d_p = (d_TP / TP_FP) if TP_FP > 0 else 0.0
    d_r = (d_TP / TP_FN) if TP_FN > 0 else 0.0
    d_f1 = (2 * d_p * d_r / (d_p + d_r)) \
        if (d_p + d_r > 0.0) else 0.0
    f1 = 0.8 * d_f1 + 0.2 * c_f1

    return {"c_p": c_p,
            "c_r": c_r,
            "c_f1": c_f1,
            "d_p": d_p,
            "d_r": d_r,
            "d_f1": d_f1,
            "f1": f1}


def _evaluate_model_bert(model, sorted_key, dataset, references, params):
    with torch.no_grad():
        model.eval()
        iterator = iter(dataset)
        counter = 0
        pad_max = 512
        
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([params.decode_batch_size, pad_max]).long()
                  for _ in range(dist.get_world_size())]
        results = []

        while True:
            try:
                features = next(iterator)
                features = lookup(features, "infer", params)
                batch_size = features["source"].shape[0]
            except:
                features = {
                    "source": torch.ones([1,1]).long(),
                    "source_mask": torch.ones([1,1]).float()
                }
                batch_size = 0
                
            t = time.time()
            counter += 1

            action_probs, action_indices = argmax_encoding(model, features, params)
            pad_batch = params.decode_batch_size - action_indices.shape[0]
            pad_length = pad_max - action_indices.shape[1]
            action_indices = torch.nn.functional.pad(
                action_indices,
                (0, pad_length, 0, pad_batch),
                value=params.lookup["label"][params.label_pad.encode("utf-8")]
            )
            
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, action_indices)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue
            
            for i in range(params.decode_batch_size):
                for j in range(dist.get_world_size()):
                    n = size[j]

                    if i >= n:
                        continue

                    seq = _convert_to_string(t_list[j][i], params)
                    results.append(seq)
        
        model.train()

        references = [reference[0] for reference in references]

        if dist.get_rank() == 0:
            restored_results = []
            for idx in range(len(results)):
                restored_results.append(results[sorted_key[idx]])
            
            restored_results_correct_length = []
            for i in range(len(restored_results)):
                label_length = len(references[i])
                restored_results_correct_length.append(restored_results[i][1:1+label_length])
            
            return f_score(restored_results_correct_length, references)

        return None


def _evaluate_model(model, sorted_key, dataset, references, params):
    # Create model
    with torch.no_grad():
        model.eval()

        iterator = iter(dataset)
        counter = 0
        pad_max = 1024

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([params.decode_batch_size, pad_max]).long()
                  for _ in range(dist.get_world_size())]
        results = []

        while True:
            try:
                features = next(iterator)
                features = lookup(features, "infer", params)
                batch_size = features["source"].shape[0]
            except:
                features = {
                    "source": torch.ones([1, 1]).long(),
                    "source_mask": torch.ones([1, 1]).float()
                }
                batch_size = 0

            t = time.time()
            counter += 1

            # Decode
            # seqs, _ = beam_search([model], features, params)


            # Padding
            seqs = torch.squeeze(seqs, dim=1)
            pad_batch = params.decode_batch_size - seqs.shape[0]
            pad_length = pad_max - seqs.shape[1]
            seqs = torch.nn.functional.pad(seqs, (0, pad_length, 0, pad_batch))

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, seqs)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(params.decode_batch_size):
                for j in range(dist.get_world_size()):
                    n = size[j]
                    seq = _convert_to_string(t_list[j][i], params)

                    if i >= n:
                        continue

                    # Restore BPE segmentation
                    seq = BPE.decode(seq)

                    results.append(seq.split())

            t = time.time() - t
            logging.info("Finished batch: {} ({:.3f} sec)".format(counter, t))

    model.train()

    if dist.get_rank() == 0:
        restored_results = []

        for idx in range(len(results)):
            restored_results.append(results[sorted_key[idx]])

        return bleu(restored_results, references)
    
    return 0.0


def evaluate(model, sorted_key, dataset, base_dir, references, params):
    if not references:
        return

    base_dir = base_dir.rstrip("/")
    save_path = os.path.join(base_dir, "eval")
    record_name = os.path.join(save_path, "record")
    log_name = os.path.join(save_path, "log")
    max_to_keep = params.keep_top_checkpoint_max

    if dist.get_rank() == 0:
        # Create directory and copy files
        if not os.path.exists(save_path):
            logging.info("Making dir: {}".format(save_path))
            os.makedirs(save_path)

            params_pattern = os.path.join(base_dir, "*.json")
            params_files = glob.glob(params_pattern)

            for name in params_files:
                new_name = name.replace(base_dir, save_path)
                shutil.copy(name, new_name)

    # Do validation here
    global_step = get_global_step()

    if dist.get_rank() == 0:
        logging.info("Validating model at step {}".format(global_step))

    score = _evaluate_model_bert(model, sorted_key, dataset, references, params)

    # Save records
    if dist.get_rank() == 0:
        logging.info("Score at step {}: c_f1: {:.4f}, d_f1: {:.4f}, f1: " \
            "{:.4f}".format(global_step, score["c_f1"], \
                score["d_f1"], score["f1"]))

        scalar("Correction Precision", score['c_p'], global_step, write_every_n_steps=1)
        scalar("Correction Recall", score['c_r'], global_step, write_every_n_steps=1)
        scalar("Correction F1", score['c_f1'], global_step, write_every_n_steps=1)
        scalar("Detection Precision", score['d_p'], global_step, write_every_n_steps=1)
        scalar("Detection Recall", score['d_r'], global_step, write_every_n_steps=1)
        scalar("Detection F1", score['d_f1'], global_step, write_every_n_steps=1)
        scalar("F1", score['f1'], global_step, write_every_n_steps=1)


        # Save checkpoint to save_path
        save({"model": model.state_dict(), "step": global_step}, save_path)

        _save_log(log_name, ("F1", global_step, score["f1"]))
        records = _read_score_record(record_name)
        record = [latest_checkpoint(save_path).split("/")[-1], score["f1"]]

        added, removed, records = _add_to_record(records, record, max_to_keep)

        if added is None:
            # Remove latest checkpoint
            filename = latest_checkpoint(save_path)
            logging.info("Removing {}".format(filename))
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        if removed is not None:
            filename = os.path.join(save_path, removed)
            logging.info("Removing {}".format(filename))
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        _save_score_record(record_name, records)

        best_score = records[0][1]
        logging.info("Best score at step " \
            "{}: {:.4f}".format(global_step, best_score))
