# README

## Population Based Training (PBT)  
Implementation of the paper [Jaderberg et al. "Population Based Training of Neural Networks"](https://arxiv.org/abs/1711.09846) (DeepMind).
PBT is an automatic method for finding a schedule of hyperparameter settings rather than a single fixed 
set of hyperparameters. A defined number of workers are initialized randomly and 
are trained in parallel for a 
certain number of epochs, after which tests are performed. 20% of workers with best 
performance are selected and their weights and hyperparameters are  copied 
to the worst 20% of workers and they are multiplied by a random mutation factor. 
There is a defined number of such training sessions.

Here PBT is applied to training an mnist classifier.

## Dependencies
- Python 3
- PyTorch 0.3

Training and testing were performed on Ununtu 16.04, Cuda 8.0 and two 1080Ti GPUs.



## Usage

### Downloads
- Clone the code
```
git clone https://github.com/victoriamazo/pbt.git
```
- Download [mnist dataset](https://drive.google.com/open?id=1_mOZwOuuMHF7Ihzrrb30RdAfgnOspHQN)
- Download the best pretrained vanilla model with fully connected layers 
[here](https://drive.google.com/open?id=17bDCJRXh8SSupTFdehI1qX5h1AYFD1ec) 
- Download the best pretrained vanilla model with convolutional layers 
[here](https://drive.google.com/open?id=1OzW-Irh_LWbqYvSsDD5FZdvJvKI2-KYq)
- Download a pretrained PBT model with fully connected layers 
 [here](https://drive.google.com/open?id=1hlULxaPINOqpZoaTIKw83T3EUNdbipty)
- Download a pretrained PBT model with convolutional layers  
[here](https://drive.google.com/open?id=1TcTZr7IxCzarZ-tsV68tWSAhAZ8dul81)

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
When running test it is recommended that args_test.json file is in a training directory,
since arguments are uploaded from the file. 



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
run in parallel first and then the rest of the workers run in parallel. 
*"num_workers_paral"* cannot be greater than *"num_worker_tot"*.


## PBT Results

In the example of hyperparameter search below (convolutional network) 
the search started with random initial 
parameters (learning rate and keep dropout probability). Workers were trained for 50 epochs
and after each epoch tests were performed. At the last epoch almost all workers 
reached highest test accuracy. An interesting conclusion is that the best learning
rate is around 0.001 and dropout probability value does not really matter.    
![alt-text-1](https://github.com/victoriamazo/pbt/blob/master/images/results.png "title-1") 

This is an example of a learning rate schedule vs. epochs.
![alt-text-2](https://github.com/victoriamazo/pbt/blob/master/images/lr.png "title-2") 