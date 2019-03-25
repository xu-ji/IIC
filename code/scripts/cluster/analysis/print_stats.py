from __future__ import print_function

import argparse
import os
import pickle

import matplotlib
import numpy as np

matplotlib.use('Agg')
"""
  Print example images 
"""

# Options ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_inds", type=int, required=True, nargs="+")
parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
given_config = parser.parse_args()

for model_ind in given_config.model_inds:
  print("Model %d ---------------- " % model_ind)
  out_dir = os.path.join(given_config.out_root, str(model_ind))
  reloaded_config_path = os.path.join(out_dir, "config.pickle")
  print("Loading restarting config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)

  # print stats_dict for best acc
  best_i = np.argmax(np.array(config.epoch_acc))
  print("best acc %s" % config.epoch_acc[best_i])
  print("average subhead acc %s" % config.epoch_avg_subhead_acc[best_i])
  print(config.epoch_stats[best_i])
