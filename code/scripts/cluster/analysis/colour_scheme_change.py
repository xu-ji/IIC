import argparse
import os
from colorsys import hsv_to_rgb

import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--file_pattern", type=str, required=True)
parser.add_argument("--file_indices", type=int, nargs="+", required=True)
parser.add_argument("--out_subdir", type=str, default="colour_change")

args = parser.parse_args()


def change_colours(img, input_colours, colours):
  h, w, c = img.shape
  assert (c == 3)
  assert (img.dtype == np.uint8)
  new_img = np.copy(img)

  for i, in_c in enumerate(input_colours):
    out_c = colours[i]
    in_c_np = np.array(in_c).reshape((1, 1, 3))
    new_img[(img == in_c_np).sum(axis=2) == 3] = out_c

  return new_img


N = 10

hues = torch.linspace(0.0, 1.0, N + 1)[0:-1]  # ignore last one
input_colours = [list((np.array(hsv_to_rgb(hue, 0.5, 0.8)) * 255.).astype(
  np.uint8)) for hue in hues]

# Colour schemes
scheme = []

# basic colours
colours = [
  [0, 0, 0],
  [177, 177, 177],
  [250, 0, 0],
  [0, 250, 0],
  [0, 0, 250],
  [250, 250, 0],
  [250, 0, 250],
  [0, 250, 250],
  [250, 100, 0],
  [0, 100, 250]
]
scheme.append(colours)

# Add functional colours
saturations = [0.5, 0.6, 0.7, 0.8]
values = [0.5, 0.6, 0.7, 0.8]
for s in saturations:
  for v in values:
    hues = torch.linspace(0.0, 1.0, N + 1)[0:-1]  # ignore last one
    colours = [list((np.array(hsv_to_rgb(hue, s, v)) * 255.).astype(
      np.uint8)) for hue in hues]
    scheme.append(colours)

# Changes the colour scheme of certain images
out_dir = os.path.join(args.in_dir, args.out_subdir)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

for i in args.file_indices:
  fname = args.file_pattern % i
  fpath = os.path.join(args.in_dir, fname)
  print("fpath: %s..." % fpath)
  img = np.array(Image.open(fpath))

  for c, curr_colours in enumerate(scheme):
    new_img = change_colours(img, input_colours, curr_colours)

    new_img = Image.fromarray(new_img)
    new_img.save(os.path.join(out_dir, "c_%d_%s" % (c, fname)))

# make composites
for c in xrange(len(scheme)):
  fnames = [args.file_pattern % i for i in args.file_indices]
  fnames = ["c_%d_%s" % (c, fname) for fname in fnames]
  fnames = [os.path.join(out_dir, fname) for fname in fnames]

  images = map(Image.open, fnames)
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]

  new_im.save(os.path.join(out_dir, "composite_%d.png" % c))
