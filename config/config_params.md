### Configuration parameters 

- _data_dir_ - full path to the root of a data directory
- _train_dir_ - full path to a training directory (is created if does not exist) 
- _model_ - name of a network ("conv" or "fc")
- _metric_ - test metric, e.g. "accuracy_categ",
- _n_filters_ - number of layers and filters in those layers in a convolutional network, 
e.g. "16,32"
- _n_hiddens_ - number of layers and their size in a fully connected network, 
e.g. "256,256"
- _version_ - version of a train ("train_unsup") and test ("test_depth") scripts
- _gpus_ - either one or several GPUs for running train/test, e.g. "0,1"
- _lr_ - learning rate
- _decreasing_lr_epochs_ - list of epochs to decrease learning rate by 2, e.g. "15,30,45"
- _num_epochs_ - number of epochs for training 
- _num_iters_for_ckpt_ - number of iterations to save a checkpoint
- _load_ckpt_ - full path to a saved model to resume training / run a test
- _keep_prob_ - 1-dropout probability
- _weight_decay_ - weight decay for training 
- _sleep_time_sec_ - waiting time for running a test (in sec) in a train/test paralellel mode 


### PBT additional configuration parameters

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

