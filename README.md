# Machine Learning Models that Remember Too Much
This repo contains an example for attacks in the paper Machine Learning that Remember Too Much (https://arxiv.org/pdf/1709.07886.pdf). 

## Added lfw support
To start a training process use 

```python train_lfw.py```

you can limit the process with ```--limit 1000``` to make it quicker while debugging 

### List of work
- understand paper
- understand this repo
- understand attacks
- refactor code to make it working after some years :)
- implement load lfw
    - re-using sklearn and changing some stuff because of shuffling
    - use gender-label data as described here: https://arxiv.org/abs/1706.04277
    - preprocess
        - mean-center
        - scale [0,1]
- TODO: create cnn network in `net.py` as proposed in the paper (currently just resnet part is re-used without upscaling and residual connections)
- TODO: create some results and interpret
    - do this for all attacks and for 1-2 parametrizations each?