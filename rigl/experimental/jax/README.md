# Weight Symmetry Research Code
This code is mostly written by Yani Ioannou. 

## Summary

Due to the symmetry of most neural networks (NN), intra-layer neuron ordering is
arbitrary, meaning that any minima of the loss function of a NN has many
equivalent minima found simply by symmetric permutations of the weight vectors
within each layer. Specifically for a deep neural network (DNN) with $$L$$
layers and $$N$$ neurons per layer, there are $$N!^L$$ permutations of each
unique minima, where each unique minima is defined by the unordered set of
neurons in each layer. Putting this into context, a 6-layer fully-connected NN
with 16 neurons/layer has $$8.39\times10^{79}$$ symmetries, close to the number
of atoms in the observable universe ($$10^{80}$$).

When such a NN is trained and subsequently pruned in an unstructured manner
however, the sparse sub-network that remains lacks this symmetry, i.e. weight
symmetry is broken, and is instead restricted to a subset of the symmetries of
the original model — put simply it is lacking the vast majority of the original
error surface's minima. This may explain why some pruned DNNs are much harder to
train from scratch, and specifically why “lottery tickets”, i.e. an
initialization associated with that specific subnetwork's symmetry, are
necessary.

## Experiment Summary

There are a number of experiment drivers defined in the base directory:

### Experiment Types {#experiment-types}

random_mask
:   Random Variable Sparsity Masks
:   This experiment generates random masks of a given type (see
    [Mask Types](#mask-types)) within the *given a sparsity range*, and trains
    the models, tracking mask statistics and training details. Masks are
    generated with a random number of connections and randomly shuffled.

shuffled_mask
:   Random Fixed Sparsity Masks
:   This experiment generates random masks of a given type (see
    [Mask Types](#mask-types)) *of a fixed sparsity*, and trains the models,
    tracking mask statistics and training details. Masks are generated with a
    fixed number of connections and simply shuffled.

fixed_param
:   Train models with (approximately) fixed number of parameters, but varying
    depth/width.
:   Train models with (approximately) fixed number of parameters, but varying
    depth/width, with shuffled mask (as in shuffled_mask driver), and only the
    MNIST_FC model type.

prune
:   Simple Pruning/Training Driver
:   This experiment trains a dense model pruning either iteratively or one-shot,
    tracking mask statistics and training details.

train
:   Simple Training Driver (Without Masking/Pruning)
:   This experiment simply trains a dense model, tracking mask statistics and
    training details.

### Mask Types {#mask-types}

symmetric
:   Structured Mask.
:   The mask is a structured

random
:   Unstructured Mask.
:   The mask as a whole is a random mask of a given sparsity, with some neurons
    having fewer/more connections than others.

per-neuron
:   Unstructured Mask.
:   Each neuron has the same sparsity (# of masked connections), but is shuffled
    randomly.

per-neuron-no-input-ablation:
:   Unstructured Mask.
:   As with per-neuron, each neuron has the same sparsity, but randomly shuffled
    connections. Also at least one connection is maintained to each of the input
    neurons (i.e. the input neurons are not effectively ablated), although these
    connections are also randomly shuffled amongst the neurons of a given layer.

### Model Types {#model-types}

MNIST_FC
:   A small fully-connected model, accepting number of neurons and depth as
    parameters. No batch normalization, configurable drop-out rate (default: 0).

MNIST_CNN
:   A small convolutional model designed for MNIST, accepting number of filters
    for each layer and depth as parameters. Uses batch normalization and
    configurable drop-out rate (default: 0).

CIFAR10_CNN
:   A larger convolutional model designed for CIFAR10, accepting number of
    filters for each layer and depth as parameters. No batch normalization,
    configurable drop-out rate (default: 0).

### Dataset Types {#dataset-types}

MNIST
:   Wrapper of the Tensorflow Datasets (TFDS) MNIST dataset.

CIFAR10
:   Wrapper of the Tensorflow Datasets (TFDS) CIFAR10 dataset.

## Running Experiments

### Running on a Workstation

Train:

```shell
python -m weight_symmetry:${EXPERIMENT_TYPE}
```

## Result Processing/Analysis

### Plotting Results from a JSON Summary File

You can convert the results to a Pandas dataframe from a JSON summary file for
plotting/analysis using the example colab in `analysis/plot_summary_json.ipynb`.
