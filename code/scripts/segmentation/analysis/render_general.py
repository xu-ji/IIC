import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch

import code.archs as archs
from code.utils.cluster.cluster_eval import \
  _get_assignment_data_matches
from code.utils.cluster.transforms import sobel_process
from code.utils.segmentation.data import make_Coco_dataloaders, \
  make_Potsdam_dataloaders
from code.utils.segmentation.render import render
from code.utils.segmentation.segmentation_eval import \
  _segmentation_get_data, segmentation_eval

# Render images for segmentation models

parser = argparse.ArgumentParser()
parser.add_argument("--model_inds", type=int, nargs="+", default=[])

parser.add_argument("--net_name", type=str, default="best")

parser.add_argument("--imgs_dataloaders", type=str, nargs="+", default=["test"])

parser.add_argument("--num", type=int, default=100)

parser.add_argument("--reassess_acc", default=False, action="store_true")

parser.add_argument("--get_match_only", default=False, action="store_true")

args = parser.parse_args()
model_inds = args.model_inds
epochs = args.epochs
net_name_prefix = args.net_name
num = args.num
reassess_acc = args.reassess_acc

print("imgs_dataloaders passed:")
print(args.imgs_dataloaders)

out_root = "/scratch/shared/slow/xuji/iid_private"

for model_ind in model_inds:
  out_dir = os.path.join(out_root, str(model_ind))
  net_names = [net_name_prefix + "_net.pytorch"]

  reloaded_config_path = os.path.join(out_dir, "config.pickle")
  print("Loading restarting config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)
  assert (config.model_ind == model_ind)

  if not hasattr(config, "use_doersch_datasets"):
    config.use_doersch_datasets = False

  if "Coco" in config.dataset:
    dataloaders_train, mapping_assignment_dataloader, mapping_test_dataloader \
      = make_Coco_dataloaders(config)
    all_label_names = [
      "sky-stuff",
      "plant-stuff",
      "ground-stuff",
    ]

    if config.include_things_labels:
      all_label_names += ["person-things"]
    if config.incl_animal_things:
      all_label_names += ["animal-things"]
  elif config.dataset == "Potsdam":
    dataloaders_train, mapping_assignment_dataloader, mapping_test_dataloader \
      = make_Potsdam_dataloaders(config)
    if config.use_coarse_labels:
      all_label_names = ["roads and cars",
                         "buildings and clutter",
                         "vegetation and trees"]
    else:
      all_label_names = ["roads",
                         "buildings",
                         "vegetation",
                         "trees",
                         "cars",
                         "clutter"]

  assert (len(all_label_names) == config.gt_k)

  print("dataloader sizes: %d %d %d" % (len(dataloaders_train[0]),
                                        len(mapping_assignment_dataloader),
                                        len(mapping_test_dataloader)))

  # ------------------------------

  for imgs_dataloader_name in args.imgs_dataloaders:
    for net_name in net_names:
      print("%s %s %s" % (
        config.out_dir, imgs_dataloader_name, net_name.split(".")[0]))
      net_name_outdir = os.path.join(config.out_dir,
                                     imgs_dataloader_name,
                                     net_name.split(".")[0])
      if not os.path.exists(net_name_outdir):
        os.makedirs(net_name_outdir)

      print("doing net_name %s to %s" % (net_name, net_name_outdir))
      sys.stdout.flush()

      # load model
      net = archs.__dict__[config.arch](config)

      model_path = os.path.join(config.out_dir, net_name)
      print("getting model path %s " % model_path)
      net.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage))
      net.cuda()
      net = torch.nn.DataParallel(net)
      net.module.eval()

      if reassess_acc:
        print("... reassessing acc %s" % datetime.now())
        sys.stdout.flush()
        stats_dict = segmentation_eval(config, net,
                                         mapping_assignment_dataloader,
                                         mapping_test_dataloader,
                                         sobel=(not config.no_sobel),
                                         return_only=True,
                                         verbose=0)
        acc = stats_dict["best"]
        print("... reassessment finished, got acc %f" % acc)
        sys.stdout.flush()
        continue

      print(
        "starting to run test data through for rendering %s" % datetime.now())
      all_matches, all_accs = _get_assignment_data_matches(net,
                                                   mapping_assignment_dataloader,
                                                   config, sobel=(not config.no_sobel),
                                                   using_IR=config.using_IR,
                                                   get_data_fn=_segmentation_get_data,
                                                   just_matches=False,
                                                   verbose=1)

      head_i = np.argmax(all_accs)
      match = all_matches[head_i]
      print("got best head %d %s" % (head_i, datetime.now()))
      print("best match %s" % str(match))

      if args.get_match_only:
        exit(0)

      colour_map_raw = [(np.random.rand(3) * 255.).astype(np.uint8)
                        for _ in xrange(max(config.output_k, config.gt_k))]

      # coco: green (veg) (7, 130, 42), blue (sky) (39, 159, 216),
      # grey (road) (82, 91, 96), red (person - if used) (229, 57, 57)
      if "Coco" in config.dataset:
        colour_map_gt = [np.array([39, 159, 216], dtype=np.uint8),
                         np.array([7, 130, 42], dtype=np.uint8),
                         np.array([82, 91, 96], dtype=np.uint8),
                         np.array([229, 57, 57], dtype=np.uint8)
                         ]
      else:
        colour_map_gt = colour_map_raw

      # render first batch
      predicted_all = [0 for _ in xrange(config.gt_k)]
      correct_all = [0 for _ in xrange(config.gt_k)]
      all_all = [0 for _ in xrange(config.gt_k)]

      if imgs_dataloader_name == "test":
        imgs_dataloader = mapping_test_dataloader
      elif imgs_dataloader_name == "train":
        imgs_dataloader = mapping_assignment_dataloader
      else:
        assert (False)

      print("length of imgs_dataloader %d" % len(imgs_dataloader))

      next_img_ind = 0

      for b_i, batch in enumerate(imgs_dataloader):
        orig_imgs, flat_targets, mask = batch
        orig_imgs, flat_targets, mask = \
          orig_imgs.cuda(), flat_targets.numpy(), mask.numpy().astype(np.bool)

        if not config.no_sobel:
          imgs = sobel_process(orig_imgs, config.include_rgb,
                               using_IR=config.using_IR)
        else:
          imgs = orig_imgs

        with torch.no_grad():
          x_outs_all = net(imgs)

        x_outs = x_outs_all[head_i]
        x_outs = x_outs.cpu().numpy()
        flat_preds = np.argmax(x_outs, axis=1)
        n, h, w = flat_preds.shape

        num_imgs_curr = flat_preds.shape[0]

        reordered_preds = np.zeros((num_imgs_curr, h, w),
                                   dtype=flat_targets.dtype)
        for pred_i, target_i in match:
          reordered_preds[flat_preds == pred_i] = target_i

        assert (mask.shape == reordered_preds.shape)
        assert (flat_targets.shape == reordered_preds.shape)
        masked = np.logical_not(mask)
        reordered_preds[masked] = -1
        flat_targets[masked] = -1  # not in colourmaps, hence will be black

        assert (reordered_preds.max() < config.gt_k)
        assert (flat_targets.max() < config.gt_k)

        # print iou per class
        for c in xrange(config.gt_k):
          preds = (reordered_preds == c)
          targets = (flat_targets == c)

          predicted = preds.sum()
          correct = (preds * targets).sum()
          all = ((preds + targets) >= 1).sum()

          predicted_all[c] += predicted
          correct_all[c] += correct
          all_all[c] += all

        if next_img_ind >= num:
          print("not rendering batch")
          continue  # already rendered num
        elif next_img_ind + num_imgs_curr > num:
          relevant_inds = range(0, num - next_img_ind)
        else:
          relevant_inds = range(0, num_imgs_curr)

        orig_imgs = orig_imgs[relevant_inds, :, :, :]
        imgs = imgs[relevant_inds, :, :, :]
        flat_preds = flat_preds[relevant_inds, :, :]
        reordered_preds = reordered_preds[relevant_inds, :, :]
        flat_targets = flat_targets[relevant_inds, :, :]

        if "Coco" in config.dataset:
          # blue and red channels are swapped
          orig_imgs_swapped = torch.zeros(orig_imgs.shape,
                                          dtype=orig_imgs.dtype)
          orig_imgs_swapped[:, 0, :, :] = orig_imgs[:, 2, :, :]
          orig_imgs_swapped[:, 1, :, :] = orig_imgs[:, 1, :, :]
          orig_imgs_swapped[:, 2, :, :] = orig_imgs[:, 0, :, :]  # ignore others
          render(orig_imgs_swapped, mode="image", name=("%d_img" % model_ind),
                 offset=next_img_ind,
                 out_dir=net_name_outdir)
          render(imgs, mode="image_as_feat", name=("%d_img_feat" % model_ind),
                 offset=next_img_ind,
                 out_dir=net_name_outdir)

        elif "Potsdam" in config.dataset:
          render(orig_imgs, mode="image_ir", name=("%d_img" % model_ind),
                 offset=next_img_ind,
                 out_dir=net_name_outdir)

        render(flat_preds, mode="preds", name=("%d_raw_preds" % model_ind),
               offset=next_img_ind,
               colour_map=colour_map_raw,
               out_dir=net_name_outdir)
        render(reordered_preds, mode="preds",
               name=("%d_reordered_preds" % model_ind),
               offset=next_img_ind,
               colour_map=colour_map_gt,
               out_dir=net_name_outdir)
        render(flat_targets, mode="preds", name=("%d_targets" % model_ind),
               offset=next_img_ind,
               colour_map=colour_map_gt,
               out_dir=net_name_outdir)

        next_img_ind += num_imgs_curr

        print("... rendered batch %d, next_img_ind %d " % (b_i, next_img_ind))
        sys.stdout.flush()

      for c in xrange(config.gt_k):
        iou = correct_all[c] / float(all_all[c])
        print("class %d: name %s: pred %d correct %d all %d %f iou" %
              (c, all_label_names[c], predicted_all[c], correct_all[c],
               all_all[c], iou))
