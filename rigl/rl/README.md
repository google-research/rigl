# The State of Sparse Training in Deep Reinforcement Learning
[**Paper**] [goo.gle/sparserl-paper](https://goo.gle/sparserl-paper)
[**Video**] [goo.gle/sparserl-video](https://goo.gle/sparserl-video)

This code requires Tensorflow 2.0; therefore we need to use a separate
requirements file. Please follow the instructions below:

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

**Citation**:
```
@InProceedings{graesser22a,
  title = 	 {The State of Sparse Training in Deep Reinforcement Learning},
  author =       {Graesser, Laura and Evci, Utku and Elsen, Erich and Castro, Pablo Samuel},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {7766--7792},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/graesser22a/graesser22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/graesser22a.html},
}
```
