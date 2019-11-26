## Rigging the Lottery

In this repository we implement following dynamic sparsity strategies:

1.  SET: Implements Sparse Evalutinory Training (SET) which corresponds to
    replacing low magniture weights with new random new connections. For more
    info see https://www.nature.com/articles/s41467-018-04316-3.

2.  SNFS: Implements momentum based training with sparsity re-distribution:
    For more info see https://arxiv.org/abs/1907.04840.

3.  RigL: Our method,RigL, removes a fraction of connections based on weight
    magnitudes and activates new ones using instantaneous gradient information.
    After updating the connectivity, training continues with the updated
    network until the next update. See our ICLR submission for more details. 

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
