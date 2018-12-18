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
[here](https://drive.google.com/open?id=1owLOz0mOvmKB64N05q6OVhwdVl4dn7YE) 
- Download the best pretrained vanilla model with convolutional layers 
[here](https://drive.google.com/open?id=1Qg2yXcNb07k2aAcRNaWygI8XJeSjMtxT)
- Download a pretrained PBT model with fully connected layers  [here]()
- Download a pretrained PBT model with convolutional layers  [here]()

### Vanilla 
For a vanilla (without PBT) training and/or testing edit the parameters ("data_dir", "train_dir") in the corresponding 
config file (*config/fc.json* for a fully-connected and *config/conv.json* for a convolutional network) 
and run
- testing (update in the config file "load_ckpt" (in the test section) with a full path to a saved model)
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
To resume training, update in the config file "load_ckpt" (in the train section) with a full 
path to a saved model.



### PBT 
For a PBT training and testing as parallel threads (testing will be run every several epochs, as defined in 
the config file) edit the parameters in the corresponding config file (*config/fc_PBT.json* for 
a fully-connected and *config/conv_PBT.json* for a convolutional network) and run
```
python3 main_PBT.py config/conv_PBT.json 
```




