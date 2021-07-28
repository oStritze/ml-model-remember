# Creation params
Needed for testing

## CAP
```
python3 train_lfw.py --attack CAP --epoch 100 --mal_p 0.1 --early_stopping_tresh 100
```

## COR
```
python3 train_lfw.py --attack cor --epoch 100 --corr 1.0 --early_stopping_tresh 100
```

## SGN
```
python3 train_lfw.py --attack SGN --epoch 100 --corr 10.0 --early_stopping_tresh 100
```