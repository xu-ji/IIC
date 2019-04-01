from __future__ import print_function

import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")

given_config = parser.parse_args()

given_config.out_dir = os.path.join(given_config.out_root,
                                    str(given_config.model_ind))

reloaded_config_path = os.path.join(given_config.out_dir, "config.pickle")
print("Loading restarting config from: %s" % reloaded_config_path)
with open(reloaded_config_path, "rb") as config_f:
  config = pickle.load(config_f)

if not hasattr(config, "batchnorm_track"):
  print("adding batchnorm track")
  config.batchnorm_track = True

if not hasattr(config, "num_sub_heads"):
  print("adding num sub heads")
  config.num_sub_heads = config.num_heads

if not hasattr(config, "select_sub_head_on_loss"):
  print("adding select_sub_head_on_loss")
  config.select_sub_head_on_loss = False

if not hasattr(config, "use_doersch_datasets"):  # only needed for seg configs
  print("adding use doersch datasets")
  config.use_doersch_datasets = False

with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
  pickle.dump(config, outfile)

with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
  text_file.write("%s" % config)

# these are for backup
with open(os.path.join(config.out_dir, "best_config.pickle"), 'wb') as outfile:
  pickle.dump(config, outfile)

with open(os.path.join(config.out_dir, "best_config.txt"), "w") as text_file:
  text_file.write("%s" % config)
