# Rigging the Lottery: Making All Tickets Winners
<img src="https://github.com/google-research/rigl/blob/master/imgs/flops8.jpg" alt="80% Sparse Resnet-50" width="45%" align="middle">

**Paper**: [https://arxiv.org/abs/1911.11134](https://arxiv.org/abs/1911.11134)

**15min Presentation** [[pml4dc](https://pml4dc.github.io/iclr2020/program/pml4dc_7.html)] [[icml](https://icml.cc/virtual/2020/paper/5808)]

## Colabs for Calculating FLOPs of Sparse Models
[MobileNet-v1](https://github.com/google-research/rigl/blob/master/rigl/imagenet_resnet/colabs/MobileNet_Counting.ipynb)

[ResNet-50](https://github.com/google-research/rigl/blob/master/rigl/imagenet_resnet/colabs/Resnet_50_Param_Flops_Counting.ipynb)

## Best Sparse Models
Parameters are float, so each parameter is represented with 4 bytes. Uniform
sparsity distribution keeps first layer dense therefore have slightly larger size
and parameters. ERK applies to all layers except for 99% sparse model, in which
we set the first layer to be dense, since otherwise we observe much worse
performance.

### Extended Training Results
Performance of RigL increases significantly with extended training iterations.
In this section we extend the training of sparse models by 5x. Note that sparse
models require much less FLOPs per training iteration and therefore most of the
extended trainings cost less FLOPs than baseline dense training.

Observing improving performance we wanted to understand where the performance of sparse networks saturates. Longest training we ran had 100x training length of the original
100 epoch ImageNet training. This training costs 5.8x of the original dense training FLOPS and the resulting 99% sparse Resnet-50 achieves an impressive 68.15% test accuracy (vs 5x training accuracy of 61.86%).

| S. Distribution |  Sparsity | Training FLOPs | Inference FLOPs | Model Size (Bytes) | Top-1 Acc | Ckpt         |
|-----------------|-----------|----------------|-----------------|-------------------------------------|-----------|--------------|
| - (DENSE)       | 0         | 3.2e18         | 8.2e9           | 102.122                             | 76.8      | -            |
| ERK             | 0.8       | 2.09x          | 0.42x           | 23.683                              | 77.17     | [link](https://storage.googleapis.com/rigl/s80erk5x.tar.gz) |
| Uniform         | 0.8       | 1.14x          | 0.23x           | 23.685                              | 76.71     | [link](https://storage.googleapis.com/rigl/s80uniform5x.tar.gz) |
| ERK             | 0.9       | 1.23x          | 0.24x           | 13.499                              | 76.42     | [link](https://storage.googleapis.com/rigl/s90erk5x.tar.gz) |
| Uniform         | 0.9       | 0.66x          | 0.13x           | 13.532                              | 75.73     | [link](https://storage.googleapis.com/rigl/s90uniform5x.tar.gz) |
| ERK             | 0.95      | 0.63x          | 0.12x           | 8.399                               | 74.63     | [link](https://storage.googleapis.com/rigl/s95erk5x.tar.gz) |
| Uniform         | 0.95      | 0.42x          | 0.08x           | 8.433                               | 73.22     | [link](https://storage.googleapis.com/rigl/s95uniform5x.tar.gz) |
| ERK             | 0.965     | 0.45x          | 0.09x           | 6.904                               | 72.77     | [link](https://storage.googleapis.com/rigl/s965erk5x.tar.gz) |
| Uniform         | 0.965     | 0.34x          | 0.07x           | 6.904                               | 71.31     | [link](https://storage.googleapis.com/rigl/s965uniform5x.tar.gz) |
| ERK             | 0.99      | 0.29x          | 0.05x           | 4.354                    | 61.86     | [link](https://storage.googleapis.com/rigl/s99erk5x.tar.gz) |
| ERK             | 0.99  | 0.58x          | 0.05x           | 4.354                               | 63.89 | [link](https://storage.googleapis.com/rigl/s99erk10x.tar.gz) |
| ERK             | 0.99  | 2.32x          | 0.05x           | 4.354                               | 66.94 | [link](https://storage.googleapis.com/rigl/s99erk40x.tar.gz) |
| ERK             | **0.99**  | 5.8x          | 0.05x           | 4.354                               | **68.15** | [link](https://storage.googleapis.com/rigl/s99erk100x.tar.gz) |

We also ran extended training runs with MobileNet-v1. Again training 100x more,
we were not able saturate the performance. Training longer consistently achieved
better results.

| S. Distribution |  Sparsity | Training FLOPs | Inference FLOPs | Model Size (Bytes) | Top-1 Acc | Ckpt         |
|-----------------|-----------|----------------|-----------------|-------------------------------------|-----------|--------------|
| - (DENSE)       | 0         | 4.5e17         | 1.14e9           | 16.864                            | 72.1      | -            |
| ERK             | 0.89       | 1.39x         | 0.21x           | 2.392                             | 69.31     | [link](https://storage.googleapis.com/rigl/mbv1_s90_erk10x.tar.gz) |
| ERK             | 0.89       | 2.79x          | 0.21x         | 2.392                              | 70.63     | [link](https://storage.googleapis.com/rigl/mbv1_s90_erk50x.tar.gz) |
| Uniform         | 0.89       | 1.25x          | 0.09x           | 2.392                              | 69.28     | [link](https://storage.googleapis.com/rigl/mbv1_s90_uniform10x.tar.gz) |
| Uniform         | 0.89       | 6.25x          | 0.09x           | 2.392                              | 70.25     | [link](https://storage.googleapis.com/rigl/mbv1_s90_uniform50x.tar.gz) |
| Uniform         | 0.89       | 12.5x          | 0.09x           | 2.392                              | 70.59     | [link](https://storage.googleapis.com/rigl/mbv1_s90_uniform100x.tar.gz) |


### 1x Training Results

| S. Distribution |  Sparsity | Training FLOPs | Inference FLOPs | Model Size (Bytes) | Top-1 Acc | Ckpt         |
|-----------------|-----------|----------------|-----------------|-------------------------------------|-----------|--------------|
| ERK             | 0.8       | 0.42x          | 0.42x           | 23.683                              | 75.12     | [link](https://storage.googleapis.com/rigl/s80erk1x.tar.gz) |
| Uniform         | 0.8       | 0.23x          | 0.23x           | 23.685                              | 74.60     | [link](https://storage.googleapis.com/rigl/s80uniform1x.tar.gz) |
| ERK             | 0.9       | 0.24x          | 0.24x           | 13.499                              | 73.07     | [link](https://storage.googleapis.com/rigl/s90erk1x.tar.gz) |
| Uniform         | 0.9       | 0.13x          | 0.13x           | 13.532                              | 72.02     | [link](https://storage.googleapis.com/rigl/s90uniform1x.tar.gz) |

### Evaluating checkpoints
Download the checkpoints and run the evaluation on ERK checkpoints with the
following:

```python
python imagenet_train_eval.py --mode=eval_once --output_dir=path/to/ckpt/folder \
    --eval_once_ckpt_prefix=model.ckpt-3200000 --use_folder_stub=False \
    --training_method=rigl --mask_init_method=erdos_renyi_kernel \
    --first_layer_sparsity=-1
```

When running checkpoints with uniform sparsity distribution use `--mask_init_method=random` and `--first_layer_sparsity=0`. Set 
`--model_architecture=mobilenet_v1` when evaluating mobilenet checkpoints.

## Sparse Training Algorithms
In this repository we implement following dynamic sparsity strategies:

1.  [SET](https://www.nature.com/articles/s41467-018-04316-3): Implements Sparse
    Evalutionary Training (SET) which corresponds to replacing low magnitude
    connections randomly with new ones.

2.  [SNFS](https://arxiv.org/abs/1907.04840): Implements momentum based training
    *without* sparsity re-distribution:

3.  [RigL](https://arxiv.org/abs/1911.11134): Our method, RigL, removes a
    fraction of connections based on weight magnitudes and activates new ones
    using instantaneous gradient information.

And the following one-shot pruning algorithm:

1. [SNIP](https://arxiv.org/abs/1810.02340): Single-shot Network Pruning based 
  on connection sensitivity prunes the least salient connections before training.

We have code for following settings:
- [Imagenet2012](https://github.com/google-research/rigl/tree/master/rigl/imagenet_resnet):
  TPU compatible code with Resnet-50 and MobileNet-v1/v2.
- [CIFAR-10](https://github.com/google-research/rigl/tree/master/rigl/cifar_resnet)
  with WideResNets.
- [MNIST](https://github.com/google-research/rigl/tree/master/rigl/mnist) with
  2 layer fully connected network.

## Setup
First clone this repo.
```bash
git clone https://github.com/google-research/rigl.git
cd rigl
```

We use [Neurips 2019 MicroNet Challenge](https://micronet-challenge.github.io/)
code for counting operations and size of our networks. Let's clone the
google_research repo and add current folder to the python path.
```bash
git clone https://github.com/google-research/google-research.git
mv google-research/ google_research/
export PYTHONPATH=$PYTHONPATH:$PWD
```

Now we can run some tests. Following script creates a virtual environment and
installs the necessary libraries. Finally, it runs few tests.
```bash
bash run.sh
```

We need to activate the virtual environment before running an experiment. With
that, we are ready to run some trivial MNIST experiments.
```bash
source env/bin/activate

python rigl/mnist/mnist_train_eval.py
```

You can load and verify the performance of the Resnet-50 checkpoints
like following.
```bash
python rigl/imagenet_resnet/imagenet_train_eval.py --mode=eval_once --training_method=baseline --eval_batch_size=100 --output_dir=/path/to/folder --eval_once_ckpt_prefix=s80_model.ckpt-1280000 --use_folder_stub=False
```

We use the [Official TPU Code](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)
for loading ImageNet data. First clone the
tensorflow/tpu repo and then add models/ folder to the python path.
```bash
git clone https://github.com/tensorflow/tpu.git
export PYTHONPATH=$PYTHONPATH:$PWD/tpu/models/
```

## Other Implementations
- [Graphcore-TF-MNIST](https://github.com/graphcore/examples/tree/master/applications/tensorflow/dynamic_sparsity/mnist_rigl): with sparse matrix ops!
- [Pytorch implementation](https://github.com/McCrearyD/rigl-torch) by Dyllan McCreary.
- [Micrograd-Pure Python](https://evcu.github.io/ml/sparse-micrograd/): This is
a toy example with pure python sparse implementation. Caution, very slow but fun.

## Citation
```
@incollection{rigl,
 author = {Evci, Utku and Gale, Trevor and Menick, Jacob and Castro, Pablo Samuel and Elsen, Erich},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {471--481},
 title = {Rigging the Lottery: Making All Tickets Winners},
 year = {2020}
}
```
## Disclaimer
This is not an official Google product.
