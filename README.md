# CapsNet
This repository contains Tensorflow implementation of Capsule Network with dynamic routing between capsules and follows the paper ["Dynamic Routing Between Capsules"](https://arxiv.org/abs/1710.09829). Note that this implementation makes extensive use of Keras as high level Tensorflow interface. Paper was followed as closely as possible, however some details of that work were not revealed. These include value of decay rate and exponential decay formula. Three common variants are:
1) lr*(decay ^ epoch) <- used in this implementation
2) lr*(e ^ (-k * step)))
3) lr*(decay_rate ^ (step / decay_steps)) <- Tensorflow provides this formulation (decay_rate and decay_steps are parameters)

Currently, no visualization code is present and this repository will change including adding tests on other datasets and pretrained models.

There are many open source implementations of CapsNet, some that I've looked at are listed bellow:
* [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)
* [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)
* [Intel's tutorial site](https://software.intel.com/en-us/articles/understanding-capsule-network-architecture)
