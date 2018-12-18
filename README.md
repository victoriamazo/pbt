# README

## Population Based Training (PBT)  
Implementation of the paper [Jaderberg et al. "Population Based Training of Neural Networks"](https://arxiv.org/abs/1711.09846) (DeepMind).
PBT is an automatic method for finding a schedule of hyperparameter settings rather than a single fixed 
set of hyperparameters. A defined number of workers are trained in parallel for a 
certain number of epochs, after which tests are performed. 20% of workers with best 
performance are selected and their weights and hyperparameters are  copied 
to the worst 20% of workers and they are multiplied by a random mutation factor. 
There is a defined number of such training sessions.

Here PBT is applied to training an mnist classifier.

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
For a vanilla (without PBT) training and/or testing edit the parameters (*"data_dir", 
"train_dir"*, etc.) in the corresponding config file (*config/fc.json* for a 
fully-connected and *config/conv.json* for a convolutional network) 
and run
- training and testing as parallel threads
```
python3 main.py config/conv.json 
```
- testing (update in the config file *"load_ckpt"* (in the test section) with a 
full path to a saved model)
```
python3 main.py config/conv.json -m test
```
- training
```
python3 main.py config/conv.json -m train
```
To resume training, update in the config file *"load_ckpt"* (in the train section) with a full 
path to a saved model.



### PBT 
For a PBT training and testing edit the parameters (*"data_dir", "train_dir"*,etc.)
in the corresponding config file (*config/fc_PBT.json* for a 
fully-connected and *config/conv_PBT.json* for a convolutional network) and run
- training and testing as parallel threads
```
python3 main_PBT.py config/conv_PBT.json 
```
- testing (update in the config file *"load_ckpt"* (in the test section) with a full path to a saved model)
```
python3 main_PBT.py config/conv_PBT.json -m test
```

Configuration parameters:
- *"PBT_lr"* and *"PBT_keep_prob"* - hyperparameters starting with *PBT* are those,
    which are taken from best performing workers and mutated
- *"mutation"* - are coefficients by which the mutated hyperparameters are 
be multiplied
- *"num_epochs"* - number of epochs for one training session
- *"num_epochs_tot"* - total number of epochs models are trained
- *"num_worker_tot"* - number of workers for one training for *"num_epochs"*
- *"num_workers_paral"* - number of parallel workers. If *"num_workers_paral"* is 
smaller than *"num_worker_tot"*, then in one training session *"num_workers_paral"* 
run in parallel first and then the rest of the workers run in parallel

Example of best hyperparameters search:
![](https://github.com/victoriamazo/ptb/images/results.png)

