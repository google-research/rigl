Code accompanying the ICML 2022 paper:
"The State of Sparse Training in Deep Reinforcement Learning".

To run DQN:

```
python -m rigl.rl.tfagents.dqn_train_eval.py \
  --gin_file=rigl/rl/tfagents/configs/dqn_atari_sparsee_config.gin
```

To run PPO:

```
python -m rigl.rl.tfagents.ppo_train_eval.py \
  --gin_file=rigl/rl/tfagents/configs/ppo_mujoco_sparse_config.gin
```

To run SAC:

```
python -m rigl.rl.tfagents.sac_train_eval.py \
  --gin_file=rigl/rl/tfagents/configs/sac_mujoco_sparse_config.gin
```
