import numpy as np


def get_patches(heatmap, centre, other, patch_side):
  # extract feature squares given coordinates
  # 10 -> 5, 11 -> 5
  d = int(np.floor(patch_side / 2.0))
  d = np.array([d, d], dtype=np.int32)

  res = []
  for point in [centre, other]:
    point = point.astype(np.int32)
    start = point - d
    end_excl = point + d + 1

    res.append(heatmap[:, :, start[0]:end_excl[0], start[1]:end_excl[1]])

  return (res[0], res[1])


def pol2cart(r, phi):
  x = r * np.cos(phi)
  y = r * np.sin(phi)
  return (y, x)
