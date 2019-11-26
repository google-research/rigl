## Rigging the Lottery: Making All Tickets Winners

**Paper**: [https://arxiv.org/abs/1911.11134](https://arxiv.org/abs/1911.11134)

In this repository we implement following dynamic sparsity strategies:

1.  [SET](https://www.nature.com/articles/s41467-018-04316-3): Implements Sparse
    Evalutionary Training (SET) which corresponds to replacing low magnitude
    weights with new random new connections.

2.  [SNFS](https://arxiv.org/abs/1907.04840): Implements momentum based training
    with sparsity re-distribution:

3.  [RigL](https://arxiv.org/abs/1911.11134): Our method, RigL, removes a
    fraction of connections based on weight magnitudes and activates new ones
    using instantaneous gradient information. After updating the connectivity,
    training continues with the updated network until the next update.

And the following one-shot pruning algorithm:

1. [SNIP](https://arxiv.org/abs/1810.02340): Single-shot Network Pruning based on Connection Sensitivity,
  prunes the least salient connections before training.

We have code for following settings:
- [Imagenet2012](https://github.com/google-research/rigl/tree/master/rigl/imagenet_resnet)
  TPU compatible code with Resnet-50, MobileNet-v1/v2.
- [CIFAR-10](https://github.com/google-research/rigl/tree/master/rigl/cifar_resnet)
  with WideResNets.
- [MNIST](https://github.com/google-research/rigl/tree/master/rigl/mnist) with 
  2 layer fully connected network.

## Setup
First clone this repo.
```
git clone https://github.com/google-research/rigl.git
cd rigl
```
We use the counter used in [Neurips 2019 MicroNet Challenge](https://micronet-challenge.github.io/).
So we need the clone the google_research repo for that and change the name.
```
git clone https://github.com/google-research/google-research.git
mv google-research/ google_research/
```
Now we can run the test. Following script creates a virtual environments and
installs the necessary libraries. Then it runs few tests.
```
bash run.sh
```
We need to activate the virtual environment before running the experiment.
We also add the current folder manually to the python path. With that, we are
ready to run some trivial mnist experiments. 

```
source env/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD

python rigl/mnist/mnist_train_eval.py
```

## Disclaimer
This is not an official Google product.
