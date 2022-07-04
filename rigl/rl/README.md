Code accompanying the ICML 2022 paper:
"The State of Sparse Training in Deep Reinforcement Learning".

This code requires Tensorflow 2.0; therefore we need to use a separate
requirenments file. Please follow the instructions below:

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
virtualenv -p python3 env_sparserl
source env_sparserl/bin/activate

pip install -r rigl/rl/requirements.txt
python -m rigl.sparse_utils_test
```

Follow instructions here to install MuJoCo: https://github.com/openai/mujoco-py#install-mujoco

To run PPO:

```
python3 rigl/rl/tfagents/ppo_train_eval.py  \
--gin_file=rigl/rl/tfagents/configs/ppo_mujoco_dense_config.gin \
--root_dir=/tmp/sparserl/ --is_mujoco=True
```

To run SAC:

```
python3 rigl/rl/tfagents/sac_train_eval.py  \
--gin_file=rigl/rl/tfagents/configs/sac_mujoco_dense_config.gin \
--root_dir=/tmp/sparserl/ --is_mujoco=True
```
