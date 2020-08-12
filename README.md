# Invariant Information Clustering for Unsupervised Image Classification and Segmentation

This repository contains PyTorch code for the <a href="https://arxiv.org/abs/1807.06653">IIC paper</a>.

IIC is an unsupervised clustering objective that trains neural networks into image classifiers and segmenters without labels, with state-of-the-art semantic accuracy. 

We set 9 new state-of-the-art records on unsupervised STL10 (unsupervised variant of ImageNet), CIFAR10, CIFAR20, MNIST, COCO-Stuff-3, COCO-Stuff, Potsdam-3, Potsdam, and supervised/semisupervised STL. For example:

<img src="https://github.com/xu-ji/IIC/raw/master/paper/unsupervised_SOTA.png" alt="unsupervised_SOTA" height=350>

Commands used to train the models in the paper <a href="https://github.com/xu-ji/IIC/blob/master/examples/commands.txt">here</a>. There you can also find the flag to turn on prediction drawing for MNIST:

<img src="https://github.com/xu-ji/IIC/blob/master/paper/progression_labelled.png" alt="progression" height=200>

How to download all our trained models including plots <a href="https://github.com/xu-ji/IIC/blob/master/examples/trained_models.txt">here</a>.

How to set up the segmentation datasets <a href="https://github.com/xu-ji/IIC/blob/master/datasets/README.txt">here</a>.

# Citation
    @inproceedings{ji2019invariant,
        title={Invariant information clustering for unsupervised image classification and segmentation},
        author={Ji, Xu and Henriques, Jo{\~a}o F and Vedaldi, Andrea},
        booktitle={Proceedings of the IEEE International Conference on Computer Vision},
        pages={9865--9874},
        year={2019}
    }

# Package dependencies
Listed <a href="https://github.com/xu-ji/IIC/blob/master/package_versions.txt">here</a>. You may want to use e.g. virtualenv to isolate the environment. It's an easy way to install package versions specific to the repository that won't affect the rest of the system.

# Running on your own dataset
You can either plug our loss (paper fig. 4, <a href="https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py#L6">here</a> and <a href="https://github.com/xu-ji/IIC/blob/master/code/utils/segmentation/IID_losses.py#L86">here</a>) into your own code, or change scripts in this codebase. Auxiliary overclustering makes a large difference (paper table 2) and is easy to implement, so it's strongly recommend even if you are using your own code; the others settings are less important.

New: single function to create the dataloaders for clustering <a href="https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/data.py#L502">here</a>.

Old thread <a href="https://github.com/xu-ji/IIC/issues/8">here</a>.


# Forks
There are various forks of the main repository. In general I have not verified the code or performance, but check them out as someone may be working with versions of interest to you. For example:
- https://github.com/astirn/IIC (Tensorflow)
- https://github.com/nathanin/IIC (Tensorflow)
- https://github.com/sebastiani/IIC (Python 3, Pytorch 1.0)
