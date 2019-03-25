from torch.utils.data.dataset import Dataset
from torchvision import transforms

from code.utils.cluster.transforms import custom_greyscale_to_tensor


class TenCropAndFinish(Dataset):
  def __init__(self, base_dataset, input_sz=None, include_rgb=None):
    super(TenCropAndFinish, self).__init__()

    self.base_dataset = base_dataset
    self.num_tfs = 10
    self.input_sz = input_sz
    self.include_rgb = include_rgb

    self.crops_tf = transforms.TenCrop(self.input_sz)
    self.finish_tf = custom_greyscale_to_tensor(self.include_rgb)

  def __getitem__(self, idx):
    orig_idx, crop_idx = divmod(idx, self.num_tfs)
    img, target = self.base_dataset.__getitem__(orig_idx)  # PIL image

    img = img.copy()

    img = self.crops_tf(img)[crop_idx]
    img = self.finish_tf(img)

    return img, target

  def __len__(self):
    return self.base_dataset.__len__() * self.num_tfs
