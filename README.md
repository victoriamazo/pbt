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
- PyTorch 1.0

Training and testing were performed on Ununtu 16.04, Cuda 8.0 and two 1080Ti GPUs.



## Usage

### Downloads
- Clone the code
```
git clone https://github.com/victoriamazo/pbt.git
```
- Download [mnist dataset](https://drive.google.com/open?id=1_mOZwOuuMHF7Ihzrrb30RdAfgnOspHQN)
- Download a pretrained fully connected model 
[here](https://drive.google.com/open?id=17bDCJRXh8SSupTFdehI1qX5h1AYFD1ec) 
- Download a pretrained convolutional model 
[here](https://drive.google.com/open?id=1OzW-Irh_LWbqYvSsDD5FZdvJvKI2-KYq)
- Download a pretrained fully connected model trained with PBT 
[here](https://drive.google.com/open?id=1hlULxaPINOqpZoaTIKw83T3EUNdbipty)
- Download a pretrained convolutional model trained with PTB 
[here](https://drive.google.com/open?id=1TcTZr7IxCzarZ-tsV68tWSAhAZ8dul81)


### Vanilla 
For a vanilla (without PBT) training and/or testing edit the config file 
(*config/fc.json* for a fully-connected and *config/conv.json* for a convolutional network, all configuration parameters are 
explained in "config/config_params.md") and run
- training and testing as parallel threads 
```
python3 main.py config/conv.json 
```
- testing
```
python3 main.py config/conv.json -m test
```
- training
```
python3 main.py config/conv.json -m train
```
When running test it is recommended that args_test.json file is in a training directory,
since arguments are uploaded from the file. 


### PBT 
For a PBT training and testing edit the corresponding config file 
(*config/fc_PBT.json* for a fully-connected and *config/conv_PBT.json* for a 
convolutional network) and run
- training and testing as parallel threads 
```
python3 main_PBT.py config/conv_PBT.json 
```
- testing 
```
python3 main_PBT.py config/conv_PBT.json -m test
```

## Results

In the example of PBT hyperparameter search below (convolutional network) 
the search started with random initial 
parameters (learning rate and keep dropout probability). Workers were trained for 50 epochs
and after each epoch tests were performed. At the last epoch almost all workers 
reached highest test accuracy. An interesting conclusion is that the best learning
rate is around 0.001 and dropout probability value does not really matter.    
![alt-text-1](https://github.com/victoriamazo/pbt/blob/master/images/results.png "PBT test accuracy") 

This is an example of a learning rate schedule (of one worker) vs. epochs.
![alt-text-2](https://github.com/victoriamazo/pbt/blob/master/images/lr.png "Learning rate") 