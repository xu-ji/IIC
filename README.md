# Invariant Information Clustering for Unsupervised Image Classification and Segmentation

This repository contains PyTorch code for the <a href="https://arxiv.org/abs/1807.06653">IIC paper</a>.

IIC is an unsupervised clustering objective that trains neural networks into image classifiers and segmenters without labels, with state-of-the-art semantic accuracy. 

We set 9 new state-of-the-art records on unsupervised STL10 (unsupervised variant of ImageNet), CIFAR10, CIFAR20, MNIST, COCO-Stuff-3, COCO-Stuff, Potsdam-3, Potsdam, and supervised/semisupervised STL. For example:

<img src="https://github.com/xu-ji/IIC/raw/master/paper/unsupervised_SOTA.png" alt="unsupervised_SOTA" height=350>

Commands used to train the models in the paper <a href="https://github.com/xu-ji/IIC/blob/master/examples/commands.txt">here</a>. There you can also find the flag to turn on prediction drawing for MNIST:

<img src="https://github.com/xu-ji/IIC/blob/master/paper/progression_labelled.png" alt="progression" height=200>

How to download all our trained models <a href="https://github.com/xu-ji/IIC/blob/master/examples/trained_models.txt">here</a>.

How to set up the segmentation datasets <a href="https://github.com/xu-ji/IIC/blob/master/datasets/README.txt">here</a>.

