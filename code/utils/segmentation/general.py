def set_segmentation_input_channels(config):
  if "Coco" in config.dataset:
    if not config.include_rgb:
      config.in_channels = 2  # just sobel
    else:
      config.in_channels = 3  # rgb
      if not config.no_sobel:
        config.in_channels += 2  # rgb + sobel
    config.using_IR = False
  elif config.dataset == "Potsdam":
    if not config.include_rgb:
      config.in_channels = 1 + 2  # ir + sobel
    else:
      config.in_channels = 4  # rgbir
      if not config.no_sobel:
        config.in_channels += 2  # rgbir + sobel

    config.using_IR = True
  else:
    raise NotImplementedError
