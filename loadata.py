#!python
#!/usr/bin/env python
from scipy.io import loadmat
from glob import glob
import os.path as osp

root = '/home/sarah/DiffSeg-Data/'

label_names = {}
with open("/home/sarah/IIC/code/datasets/segmentation/FreeSurferColorLUT.txt") as f:
    for line in f:
        vals = line.split()
        if len(vals) > 2 and vals[0].isdigit():
            label_names[vals[0]] = vals[1]

import csv

w = csv.writer(open(osp.join(root, "labelNameCount.csv"), "w"))
index = 0
with open(osp.join(root, "labels.csv")) as label_counts:
    reader = csv.reader(label_counts)
    for rows in reader:
        label = rows[0]
        count = rows[1]
        name = label_names[label]
        w.writerow([label, index, count, name])
        index += 1

# subjects = sorted(glob(osp.join(root, 'mwu*')))

# actual_labels = {}
# for subject_id in subjects:
#     image_mat = loadmat(osp.join(root, subject_id, "data.mat"))
#     for s in range(image_mat['segs'].shape[2]):
#         label = image_mat['segs'][:, :, s, 1]
#         for i in range(len(label)):
#             for j in range(len(label[0])):
#                 if label[i, j] not in actual_labels:
#                     actual_labels[label[i, j]] = 1
#                 else:
#                     actual_labels[label[i, j]] += 1

# import csv
# w = csv.writer(open(osp.join(root, "labels.csv"), "w"))
# for key, val in actual_labels.items():
#     w.writerow([key, val])

# print(len(actual_labels))
# print(actual_labels)

# import matplotlib.pyplot as plt
# f, axarr = plt.subplots(3,2)

# # plt.show()

# print(x['imgs'][:, :, slide, 1].min(), x['imgs'][:, :, slide, 1].max())
# axarr[0,0].imshow(x['imgs'][:, :, slide, 0])
# axarr[0,1].imshow(x['imgs'][:, :, slide, 1])
# axarr[1,0].imshow(x['imgs'][:, :, slide, 2])
# axarr[1,1].imshow(x['imgs'][:, :, slide, 3])
# # axarr[2,0].imshow(x['segs'][:, :, slide, 0],  cmap='plasma', vmin=0, vmax=77)
# axarr[2,0].imshow(x['segs'][:, :, slide, 1],  cmap='plasma', vmin=0, vmax=2033)
# axarr[2,1].imshow(label,  cmap='plasma')

# # plt.colorbar()
# plt.show()

# %%