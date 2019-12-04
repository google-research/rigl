# Rigging the Lottery: Making All Tickets Winners
<img src="https://github.com/google-research/rigl/blob/master/imgs/flops8.jpg" alt="80% Sparse Resnet-50" width="45%" align="middle">

**Paper**: [https://arxiv.org/abs/1911.11134](https://arxiv.org/abs/1911.11134)

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

## Disclaimer
This is not an official Google product.
