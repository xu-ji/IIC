import numpy as np
from torch.utils.data import Dataset


class TruncatedDataset(Dataset):
  def __init__(self, base_dataset, pc):
    self.base_dataset = base_dataset
    self.len = int(len(self.base_dataset) * pc)
    # also shuffles. Ok because not using for train
    self.random_order = np.random.choice(len(self.base_dataset), size=self.len,
                                         replace=False)

  def __getitem__(self, item):
    assert (item < self.len)

    return self.base_dataset.__getitem__(self.random_order[item])
    # return self.base_dataset.__getitem__(item)

  def __len__(self):
    return self.len
