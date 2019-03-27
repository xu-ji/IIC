import os
import pickle

import yaml

# first 12 are "things", next 15 are "stuff"
# arbitrary order as long as things first, then stuff
_sorted_coarse_names = [
  "electronic-things",  # 0
  "appliance-things",  # 1
  "food-things",  # 2
  "furniture-things",  # 3
  "indoor-things",  # 4
  "kitchen-things",  # 5
  "accessory-things",  # 6
  "animal-things",  # 7
  "outdoor-things",  # 8
  "person-things",  # 9
  "sports-things",  # 10
  "vehicle-things",  # 11

  "ceiling-stuff",  # 12
  "floor-stuff",  # 13
  "food-stuff",  # 14
  "furniture-stuff",  # 15
  "rawmaterial-stuff",  # 16
  "textile-stuff",  # 17
  "wall-stuff",  # 18
  "window-stuff",  # 19
  "building-stuff",  # 20
  "ground-stuff",  # 21
  "plant-stuff",  # 22
  "sky-stuff",  # 23
  "solid-stuff",  # 24
  "structural-stuff",  # 25
  "water-stuff"  # 26
]

_sorted_coarse_name_to_coarse_index = \
  {n: i for i, n in enumerate(_sorted_coarse_names)}


# full loop is run, any results from yield are collected - can return multiple
# but name only exists once in d
def _find_parent(name, d):
  for k, v in d.iteritems():
    if isinstance(v, list):
      if name in v:
        yield k
    else:
      assert (isinstance(v, dict))
      for res in _find_parent(name, v):  # if it returns anything to us
        yield res


def generate_fine_to_coarse(out_path):
  print("generating fine to coarse files in %s ..." % out_path)
  # -1 (unlabelled) is not in dict, not used (see _CocoStuff._fine_to_coarse)

  print(os.getcwd())

  with open("./code/datasets/segmentation/util/cocostuff_fine_raw.txt") as f:
    l = [tuple(pair.rstrip().split('\t')) for pair in f]
    l = [(int(ind), name) for ind, name in l]

  with open("./code/datasets/segmentation/util/cocostuff_hierarchy.y") as f:
    d = yaml.load(f)

  fine_index_to_coarse_index = {}
  fine_name_to_coarse_name = {}
  for fine_ind, fine_name in l:
    assert (fine_ind >= 0 and fine_ind < 182)
    parent_name = list(_find_parent(fine_name, d))
    # print("parent_name of %d %s: %s"% (fine_ind, fine_name, parent_name))
    assert (len(parent_name) == 1)
    parent_name = parent_name[0]
    parent_ind = _sorted_coarse_name_to_coarse_index[parent_name]
    assert (parent_ind >= 0 and parent_ind < 27)

    fine_index_to_coarse_index[fine_ind] = parent_ind
    fine_name_to_coarse_name[fine_name] = parent_name

  assert (len(fine_index_to_coarse_index) == 182)

  # write fine_index_to_coarse_index, fine_name_to_coarse_name
  print("dumping to: %s" % out_path)
  with open(out_path, "wb") as out_f:
    pickle.dump({"fine_index_to_coarse_index": fine_index_to_coarse_index,
                 "fine_name_to_coarse_name": fine_name_to_coarse_name},
                out_f)

  with open(out_path + ".txt", "w") as out_f:
    print("fine_name_to_coarse_name:")
    for k, v in fine_name_to_coarse_name.iteritems():
      out_f.write("%s\t%s" % (k, v))

    print("fine_index_to_coarse_index:")
    for k, v in fine_index_to_coarse_index.iteritems():
      out_f.write("%d\t%d" % (k, v))
