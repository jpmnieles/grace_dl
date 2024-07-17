# Grace Robot Deep Learning Repository

## How to run

- Default
~~~
python src/train.py trainer=gpu
~~~
- Task Name: changes the saved folder name
~~~
python src/train.py task_name=sample_name
~~~
- Logger: tensorboard or wandb with settings in config files
~~~
python src/train.py logger=tensorboard
~~~
- Tags: maximum of two tags that adds explicit label file on the runs
~~~
python src/train.py tags=["tag1", "tag2"] 
~~~
- Override any parameters based on the config files
~~~
python src/train.py trainer.max_epochs=20 data.batch_size=64
~~~
- Experiment: load an experiment yaml config from configs/experiment
~~~
python src/train.py experiment=experiment_name.yaml
~~~

## Sample Experiment File
~~~
# @package _global_

# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /data: mnist.yaml
  - override /model: mnist.yaml
  - override /callbacks: default.yaml
  - override /trainer: default.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["mnist", "simple_dense_net"]

seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 0.002
  net:
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

data:
  batch_size: 64

logger:
  wandb:
    tags: ${tags}
    group: "mnist"
~~~
