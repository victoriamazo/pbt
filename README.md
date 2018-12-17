# README

## Population Based Training (PBT)  
Implementation of the paper [Jaderberg et al. "Population Based Training of Neural Networks"](https://arxiv.org/abs/1711.09846) (DeepMind).
PBT is an automatic method for finding a schedule of hyperparameter settings rather than a single fixed 
set of hyperparameters. Here PBT is applied to training an mnist classifier.

## Dependencies
- Python 3
- PyTorch 0.3

## Usage

### Downloads
- Clone the code
```
git clone https://github.com/victoriamazo/pbt.git
```
- Download [mnist dataset](https://drive.google.com/open?id=1_mOZwOuuMHF7Ihzrrb30RdAfgnOspHQN)
- Download the best pretrained vanilla model with fully connected layers  
- Download the best pretrained vanilla model with convolutional layers 
- Download a pretrained PBT model

### Vanilla 
For a vanilla (without PBT) training and/or testing edit the parameters in the corresponding 
config file (*config/fc.json* for a fully-connected and *config/conv.json* for a convolutional network) 
and run
- testing
```
python3 main.py config/conv.json -m test
```
- training
```
python3 main.py config/conv.json -m train
```
- training and testing as parallel threads
```
python3 main.py config/conv.json 
```
To resume training in a config file update the training directory "train_dir", where a pretrained model 
is located in the training directory, write the last iteration in the "load_ckpt_iter" in the 
"train" subsection.

### PBT 
For a PBT training and testing as parallel threads (testing will be run every several epochs, as defined in 
the config file) edit the parameters in the corresponding config file (*config/fc_PBT.json* for 
a fully-connected and *config/conv_PBT.json* for a convolutional network) and run
```
python3 main_PBT.py config/conv_PBT.json 
```




