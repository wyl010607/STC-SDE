# Code for STC-SDE
the code of paper STC-SDE: Trustworthy Multivariate Time Series Prediction for Industrial Process Control

The implementation of the STCSDE model can be found in the directory models/STCSDE
## Train and Test
```
python main.py
```
Model parameters can be modified in DIST_NDE.yaml, and then training parameters can be modified in STCSDE_train_config.yaml. The maximum number of iterations and the model save location are passed in by command line arguments
```
python main.py --model_name STCSDE --num_epoch 100 --model_save_path ./model_save --result_save_dir ./result_save
```
## Datasets:
[DIST datasets](https://github.com/wyl010607/DIST_Dataset)
