import torch
torchdict = torch.load("/home/yuxinh/dl_seg/IIC/out/777/latest.pytorch", map_location=lambda storage, loc: storage)
for key, value in torchdict.items():
    print(key)
    for k1,v1 in torchdict[key].items():
        print("--", k1)
