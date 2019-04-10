import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def render(data, mode, name, colour_map=None, offset=0, out_dir=""):
  if isinstance(data, torch.Tensor):
    if data.is_cuda:
      data = data.cpu()
    data = data.numpy()

  # multiple inputs
  if ("image" in mode) or mode == "label":
    if len(data.shape) == 4:
      for i in range(data.shape[0]):
        render(data[i, :, :, :], mode=mode, name=(name + "_%d" % (i + offset)),
               out_dir=out_dir)
      return
    else:
      assert (len(data.shape) == 3)
  else:
    assert (mode == "mask" or mode == "matrix" or mode == "preds")
    if len(data.shape) == 3:
      for i in range(data.shape[0]):
        render(data[i, :, :], mode=mode, name=(name + "_%d" % (i + offset)),
               colour_map=colour_map, out_dir=out_dir)
      return
    else:
      assert (len(data.shape) == 2)

  # recursively called case for single inputs
  out_handle = os.path.join(out_dir, name)

  if mode == "image":
    data = data.transpose((1, 2, 0))  # channels last
    if data.shape[2] == 4:
      # pre-sobel with rgb

      # don't render grey, only colour

      # data_grey = data[:, :, 3] * 255.
      # img_grey = Image.fromarray(data_grey.astype(np.uint8))
      # img_grey.save(out_handle + "_grey.png")
      data = data[:, :, :3]
    else:
      # pre-sobel no rgb

      # render grey
      assert (data.shape[2] == 1)
      data = data.squeeze(axis=2)

    data *= 255.
    img = Image.fromarray(data.astype(np.uint8))
    img.save(out_handle + ".png")

  elif mode == "image_ir":
    data = data.transpose((1, 2, 0))  # channels last
    if data.shape[2] == 5:  # rgb, grey, ir
      # pre-sobel with rgb
      # don't render grey, only colour
      data = data[:, :, :3]
    elif (data.shape[2] == 2):  # grey, ir
      # pre-sobel no rgb
      # render grey
      data = data[:, :, 0]
    elif (data.shape[2] == 3) or (data.shape[2] == 4):  # no sobel
      data = data[:, :, :3]

    data *= 255.
    img = Image.fromarray(data.astype(np.uint8))
    img.save(out_handle + ".png")

  elif mode == "image_as_feat":
    data = data.transpose((1, 2, 0))
    if data.shape[2] == 5:
      # post-sobel with rgb

      # only render sobel

      data_sobel = data[:, :, [3, 4]].sum(axis=2, keepdims=False) * 0.5 * 255.
      img_sobel = Image.fromarray(data_sobel.astype(np.uint8))
      img_sobel.save(out_handle + ".png")
      return

      data = data[:, :, :3]
    elif data.shape[2] == 2:
      # post_sobel no rgb

      # only render sobel

      data = data.sum(axis=2, keepdims=False) * 0.5
    else:
      assert (False)

    data *= 255.
    img = Image.fromarray(data.astype(np.uint8))
    img.save(out_handle + ".png")

  elif mode == "mask":
    # only has 1s and 0s, whatever the dtype
    img = Image.fromarray(data.astype(np.uint8) * 255)
    img.save(out_handle + ".png")

  elif mode == "label":
    # render histogram, with title (if current labels contains 0-11, 12-26,
    # 0-91, 92-181)
    assert (data.dtype == np.int32 or data.dtype == np.int64)

    # 0 (-1), [1 (0), 12 (11)], [13 (12), 27 (26)]
    hist = _make_hist(data)
    inds = np.nonzero(hist > 0)[0]
    min_ind = inds.min()
    max_ind = inds.max()

    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.plot(hist)
    ax.set_title("Labels for %s, min %s, max %s" % (name, min_ind, max_ind))
    fig.canvas.draw_idle()
    fig.savefig(out_handle + ".png")

  elif mode == "matrix":
    with open(out_handle + ".txt", "w") as f:
      f.write(str(data))

  elif mode == "preds":
    h, w = data.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # ignore <0 labels
    for c in range(0, data.max() + 1):
      img[data == c, :] = colour_map[c]

    img = Image.fromarray(img)
    img.save(out_handle + ".png")

  else:
    assert (False)


def _make_hist(tensor):
  res = np.zeros(183)
  for i in range(-1, 181 + 1):
    res[i + 1] = (tensor == i).sum()

  return res
