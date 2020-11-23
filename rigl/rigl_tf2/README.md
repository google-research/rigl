# Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win
<img src="https://github.com/google-research/rigl/blob/master/imgs/lottery_init.jpg" alt="Lottery Tickets explained" width="80%" align="middle">
**Paper**: [https://arxiv.org/abs/2010.03533](https://arxiv.org/abs/2010.03533)

This code includes a TF-2 implementation of RigL and some other popular sparse training methods along with pruning, scratch and lottery ticket experiments in a unified codebase.


Run pruning experiments.

```
python train.py --gin_config=configs/prune.gin
```

Runs lottery training.

```
Lottery experiments:
python train.py logdir=/tmp/sparse_spectrum/lottery --seed=8 \
--gin_config=configs/lottery.gin
```

Runs scratch training.

```
python train.py --logdir=/tmp/sparse_spectrum/scratch --seed=8 \
--gin_config=configs/scratch.gin
```

For assigning different gin flags use gin_bindings. i.e.

```
`--gin_bindings='network.weight_init_method="unit_scaled"'
--gin_bindings='unit_scaled_init.init_method="faninout_uniform"'
```

Calculating eigenvalues of hessian. Use logdir to point different checkpoints.

```
python train.py --mode=hessian \
--gin_config=configs/hessian.gin
```

Point `mlp_configs` to run MLP experiments.

```
python train.py  --gin_config=mlp_configs/prune.gin
```

Running interpolation experiments is done as the following:

```
python interpolate.py --logdir=/tmp/sparse_spectrum/scratch \
--gin_config=configs/interpolate.gin \
--ckpt_start=/path_to_lottery_logdir/cp-11719.ckpt \
--ckpt_end=/path_to_prune_logdir/cp-11719.ckpt \
--operative_gin=/path_to_logdir/operative_config.gin \
--logdir=/path_to_prune_logdir/ltsolution2prune/
```

## a journey with train.py.

1) check `main()`.

-   Load preload_gin_config. This is useful for scratch experiments to use same
    hyper_parameters as the pruning experiments. We can overwrite these with
    regular `gin_configs/bindings` flags.
-   Load data and create the network. Network might load its values from a
    checkpoint. These arguments are set through gin. See utils.get_network for
    details.
-   Then the code either trains the network `mode=train_eval` or calculates the
    hessian: `mode=hessian`.

2) train_model()

-   Create the optimizer and samples a validation set from the training set.
    Validation set is a subset of the training set and used to get better
    estimates of certain metrics.
-   Create the `mask_updater` object. The returned value can be none, then the
    masks are not updated.
-   Perform pre-training updates to the network: i.e. meta_initialization.
-   Set-up checkpointing so that if a checkpoint exist continue from where it is
    left.
-   Define gradient function. This function is used during training and for
    certain other metrics. Note that we have to manually mask the gradients
    since they are dense.
-   Define logging function for logging tensorboard event summaries.
-   Main training loop: save, log, gradient step, mask update.
