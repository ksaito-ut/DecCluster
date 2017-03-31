Implementation of DEC
====

Overview
This code is an Implementation of ["Unsupervised Deep Embedding for Clustering Analysis"]in tensorflow.
The paper is https://ai2-website.s3.amazonaws.com/publications/unsupervised-deep-embedding.pdf. Our implementation is different from original one.

## Description
We simply implemented the method for MNIST. We do not pretrain the model from Stacked AE. We jointly optimize the objective of the AE and clustering.

## Usage
```
python mnist_dec.py
```


## Requirement
* python-2.7
* tensorflow-0.11.0

