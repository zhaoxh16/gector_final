# coding=utf-8

from thuctc.utils.hparams import HParams
from thuctc.utils.inference import beam_search, argmax_decoding
from thuctc.utils.inference import argmax_encoding
from thuctc.utils.evaluation import evaluate
from thuctc.utils.checkpoint import save, latest_checkpoint
from thuctc.utils.scope import scope, get_scope, unique_name
from thuctc.utils.misc import get_global_step, set_global_step
from thuctc.utils.convert_params import params_to_vec, vec_to_params
