# Machine Learning Models that Remember Too Much
This repo contains an example for attacks in the paper Machine Learning that Remember Too Much (https://arxiv.org/pdf/1709.07886.pdf). 

## prepare conda space 
Use `build_conda_env.sh` to create conda env and install necessary packages to run everything.
```bash build_conda_env.sh```


## Added lfw support
To start a training process use 

```python train_lfw.py```

you can limit the process with ```--limit 1000``` to make it quicker while debugging 

### Test results

Use ```python train_lfw.py --epoch 25``` to create correlation attack model on lfw data gender recognition (~25 min runtime).  

```python test_model_lfw.py``` will access `models/lfw_cor_1.0_model.npz` by default (correlation factor $1.0$) and output iamges in `imgs/` folder for the run and a accuracy and Mean Absolute Pixel Error MAPE term on the console for the run. Other params can be used by using the run parameters for the python scripts. 

## Results
Results for our experiments can be found in `results/attacks.md`. 

---


### List of work
- understand paper
- understand this repo
- understand attacks
- refactor code to make it work after some years :)
- implement load lfw
    - re-using sklearn and changing some stuff because of shuffling
    - use gender-label data as described here: https://arxiv.org/abs/1706.04277
    - preprocess
        - mean-center
        - scale [0,1]
- create cnn network in `net.py` as proposed in the paper (currently just resnet part is re-used without upscaling and residual connections)
- TODO: create some results and interpret
    - do this for all attacks and for 1-2 parametrizations each?