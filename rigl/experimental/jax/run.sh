# Copyright 2022 RigL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r weight_symmetry/requirements.txt
TEST_NAMES='training.training_test
train_test
fixed_param_test
shuffled_mask_test
models.model_factory_test
models.cifar10_cnn_test
models.mnist_cnn_test
models.mnist_fc_test
utils.utils_test
prune_test
random_mask_test
pruning.mask_factory_test
pruning.init_test
pruning.symmetry_test
pruning.pruning_test
pruning.masked_test
datasets.dataset_factory_test
datasets.dataset_base_test
datasets.cifar10_test
datasets.mnist_test'

IFS=$'\n' readarray -t tests <<<$TEST_NAMES

for test in ${tests[@]}; do
  python3 -m "weight_symmetry.${test}"
done
