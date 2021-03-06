'''
Main allows to run Population Based Training (PBT), where trainings and tests run as separate processes.
Configuration should be specified in the config json file, given as an argument.
When running test it is recommended that args_test.json file is in a training directory,
since arguments are uploaded from the file.

Examples:
* trains and tests (PBT):
    config/conv_PBT.json
* tests:
    config/conv_PBT.json -m test

By adding '--debug', no tensorboard will start
'''


import matplotlib
matplotlib.use('Agg')
import argparse
import json
import multiprocessing
import os
import math
from argparse import Namespace
import numpy as np
import time
import shutil
from time import sleep
import pandas as pd
from collections import OrderedDict
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = '1'
from trains.train_builder import Train
from tests.test_builder import Test
from utils.visualization import w_params_visualization


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='DIR', help="(required) Path to config file")
    parser.add_argument('-m', '--mode', type=str, default='', help="(optional) 'test'")
    parser.add_argument('--debug', action='store_true', help='Deactivation of tensorboard and csv writers for debugging')
    args = parser.parse_args()
    return args


def input_params(args=None):
    # get args, check them and write them to json file
    if args is None:
        args = getArgs()
    mode = 'PBT_training'
    if args.mode != '':
        mode = args.mode
    debug = False
    if args.debug:
        debug = True
    config_filename = args.config

    present_dir = os.getcwd()
    config_path = os.path.join(present_dir, config_filename)
    with open(config_path, 'rt') as r:
        config = json.load(r)
    config_general = config['general']
    root_dir = config_general['train_dir']
    exp_time = str(time.strftime('%y%m%b%d_%H-%M-%S', time.localtime(time.time())))
    config_general['train_dir'] = os.path.join(config_general['train_dir'], exp_time)

    config_train = config['train']
    config_train.update(config_general)
    config_test = config['test']
    config_test.update(config_general)
    config_val = {}
    if 'val' in config:
        config_val = config['val']
        config_val.update(config_general)

    config_train['debug'] = debug
    config_test['debug'] = debug
    if 'val' in config:
        config_val['debug'] = debug
    return config_train, config_test, config_val, mode, root_dir


def set_cuda_visible_gpus(config, mode, one_gpu_str=None):
    if 'gpus' in config and len(config['gpus']) > 0:
        gpus_list = config['gpus']
        gpus_list = list(map(int, gpus_list.split(',')))
        gpu_list = ','.join([str(i) for i in gpus_list])
        if one_gpu_str != None:
            os.environ["CUDA_VISIBLE_DEVICES"] = one_gpu_str
            print('running {} on gpu {}'.format(mode, one_gpu_str))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
            print('running {} on gpu(s) {}'.format(mode, gpu_list))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''             # run on cpu
        gpu_list = []
        print('running {} on cpu'.format(mode))
    return gpu_list


def run_one_epoch(PBT_params, PBT_params_worker, num_workers, config_train, config_test, worst_20perc_workers,
                  best_20perc_workers, epoch_loop_num, num_worker_loops):
    # initiate some fields in PBT_params dictionary
    for param, value in PBT_params.items():
        PBT_params_worker['mutation_{}'.format(param)] = (-1) * np.ones(num_workers)
    PBT_params_worker['copied_from_w'] = (-1) * np.ones(num_workers)
    num_workers_paral = int(config_train['num_workers_paral'])
    gpus_list = list(map(int, config_train['gpus'].split(',')))
    gpus_list_paral = gpus_list * int(math.ceil(num_workers_paral / float(len(gpus_list))))
    w = 0
    for worker_loop_num in range(num_worker_loops):
        for worker_paral in range(num_workers_paral):
            print('\nepoch_loop_num = {}, w = {}, worker_loop_num = {}, worker_paral = {}'.format(epoch_loop_num, w,
                                                                    worker_loop_num, worker_paral))

            if epoch_loop_num == 0:
                # rand initial params
                for param, value in PBT_params.items():
                    assert len(value) > 1
                    if len(value) == 2:
                        val = np.random.uniform(value[0], value[1])
                    else:
                        idx = np.random.randint(0, len(value))
                        val = value[idx]
                    config_train[param] = val
                    PBT_params_worker[str(param)][w] = val
            else:
                # copy all checkpoints
                ckpt_dir = os.path.join(config_train['train_dir'], 'ckpts')
                for filename in os.listdir(ckpt_dir):
                    if filename.startswith('latest_'):
                        src = os.path.join(ckpt_dir, filename)
                        ww = (filename.split('.')[0]).split('_')[-1]
                        dest = os.path.join(ckpt_dir, 'latestcopy_{}.pth'.format(ww))
                        if os.path.isfile(dest):
                            os.remove(dest)
                        shutil.copy(src, dest)

                # truncation selection (copy weights and params of best 20% workers randomly to worst 20% workers)
                assert best_20perc_workers != None and worst_20perc_workers != None
                if w in worst_20perc_workers:
                    rand_idx = np.random.randint(0, len(best_20perc_workers))
                    worker_to_copy_from = best_20perc_workers[rand_idx]
                    PBT_params_worker['copied_from_w'][w] = worker_to_copy_from
                    print('w = {}, best_20perc_workers = {}, worker_to_copy_from = {}'.format(
                        w, best_20perc_workers, worker_to_copy_from))
                    mutation = [float(val) for val in config_train['mutation'].split(',')]
                    for param, value in PBT_params.items():
                        rand_idx = np.random.randint(0, len(mutation))
                        mutation_rand = mutation[rand_idx]
                        val = float(PBT_params_worker[str(param)][worker_to_copy_from] * mutation_rand)
                        if param == 'keep_prob':
                            val = min(val, 1.0)
                            val = max(val, 0.0)
                        config_train[param] = val
                        # print 'worst20%: w = {}, param = {}, value = {}'.format(w, param, val)
                        PBT_params_worker[str(param)][w] = val
                        PBT_params_worker['mutation_{}'.format(param)][w] = mutation_rand
                    # resume training from the ckpt of the best worker
                    config_train['load_ckpt'] = 'latest_{}'.format(worker_to_copy_from)
                else:
                    # resume training from previous ckpt
                    config_train['load_ckpt'] = 'latest_{}'.format(w)

            # add current epoch number to config
            config_train['n_epoch'] = epoch_loop_num * config_train['num_epochs']

            # start train process
            config_train['worker_num'] = w
            print('started training worker {}'.format(w))
            set_cuda_visible_gpus(config_train, 'train', one_gpu_str=str(gpus_list_paral[worker_paral]))
            train_process = multiprocessing.Process(target=Train.train_builder, args=(Namespace(**config_train),),
                                                    name='train_{}'.format(w))
            train_process.start()
            w += 1

        train_process.join()
        print('worker loop {} for epoch_loop {} finished'.format(worker_loop_num, epoch_loop_num))
        sleep(5)

        # run parallel tests for all workers
        w = 0
        for worker_loop_num in range(num_worker_loops):
            for worker_paral in range(num_workers_paral):
                print('started test worker {}'.format(w))
                config_test['worker_num'] = w
                set_cuda_visible_gpus(config_test, 'test', one_gpu_str=str(gpus_list_paral[worker_paral]))
                test_process = multiprocessing.Process(target=Test.test_builder, args=(Namespace(**config_test),),
                                                       name='test_{}'.format(w))
                test_process.start()
                w += 1
            test_process.join()
        print('test for epoch_loop {} finished'.format(epoch_loop_num))
        sleep(5)


def read_accuracies_and_fill_history_csv(train_dir, num_workers, PBT_params, PBT_params_worker, epoch):
    # read and rank accuracies of all workers
    results_dict = {}
    test_iter = 0
    for w in range(num_workers):
        results_table_path = os.path.join(train_dir, 'results_{}.csv'.format(str(w)))
        if os.path.isfile(results_table_path):
            results_table = pd.read_csv(results_table_path, index_col=0)
            if len(results_table.index) > 0:
                row_idx = len(results_table.index) - 1
                test_iter = int(results_table.iloc[row_idx]['iter'])
                test_acc = results_table.iloc[row_idx]['test_acc']
                results_dict[str(w)] = test_acc
        else:
            print('no {}'.format(results_table_path))
    # dictionary sorted by value (from worst to best accuracy)
    results_dict_sorted = OrderedDict(sorted(results_dict.items(), key=lambda t: t[1]))
    workers_sorted = []
    for w, accuracy in results_dict_sorted.items():
        workers_sorted.append(w)
    num_workers_20perc = int(math.ceil(num_workers / 5.0))
    best_20perc_workers = workers_sorted[-num_workers_20perc:]
    best_20perc_workers = [int(w) for w in best_20perc_workers]
    worst_20perc_workers = workers_sorted[:num_workers_20perc]
    worst_20perc_workers = [int(w) for w in worst_20perc_workers]
    print('results_dict_sorted = ', results_dict_sorted)
    print('best_20perc_workers = {}, worst_20perc_workers = {}\n\n'.format(best_20perc_workers, worst_20perc_workers))

    # copy hyperparams and accuracies to history.csv
    history_path = os.path.join(train_dir, 'history.csv')
    row_idx = 0
    if not os.path.isfile(history_path):
        columns = ['worker', 'epoch', 'iter', 'test_acc']
        index = np.arange(1)
        history_table = pd.DataFrame(columns=columns, index=index)
        for param, value in PBT_params.items():
            history_table[str(param)] = None
    else:
        history_table = pd.read_csv(history_path, index_col=0)
        if len(history_table.index) > 0:
            row_idx = len(history_table.index)
    for w in range(num_workers):
        history_table.ix[row_idx, 'worker'] = w
        history_table.ix[row_idx, 'epoch'] = epoch
        history_table.ix[row_idx, 'iter'] = test_iter
        history_table.ix[row_idx, 'test_acc'] = results_dict[str(w)]
        for param, value in PBT_params_worker.items():
            history_table.ix[row_idx, str(param)] = value[w]
        row_idx += 1
    # for param, value in PBT_params_worker.items():
    #     print 'PBT_params_worker: param {} = {}'.format(param, value)
    history_table.to_csv(history_path, index=True)

    return best_20perc_workers, worst_20perc_workers, workers_sorted


def main(args=None):
    print('running PBT training')

    # input parameters (dictionary)
    config_train, config_test, config_val, mode, root_dir = input_params(args=args)

    if mode == 'PBT_training':
        # dict of PBO params ranges
        PBT_params = {}
        for param, value in config_train.items():
            if param.startswith('PBT'):
                PBT_params[str(param[4:])] = [float(val) for val in value.split(',')]

        num_workers = int(config_train['num_worker_tot'])
        num_workers_paral = int(config_train['num_workers_paral'])
        assert num_workers_paral <= num_workers
        num_epoch_loops = int(math.ceil(int(config_train['num_epochs_tot']) / float(config_train['num_epochs'])))
        num_worker_loops = int(math.ceil(num_workers / float(num_workers_paral)))

        PBT_params_worker = {}                    # key - param name, value - list of param values for each worker
        for param, value in PBT_params.items():
            PBT_params_worker[str(param)] = np.zeros(num_workers)
            PBT_params_worker['mutation_{}'.format(param)] = (-1) * np.ones(num_workers)
        PBT_params_worker['copied_from_w'] = (-1) * np.ones(num_workers)

        # run parallel training 'num_epochs_tot/num_epochs' times
        worst_20perc_workers, best_20perc_workers = -1, -1
        for epoch_loop_num in range(num_epoch_loops):
            run_one_epoch(PBT_params, PBT_params_worker, num_workers, config_train, config_test, worst_20perc_workers,
                                        best_20perc_workers, epoch_loop_num, num_worker_loops)
            # read and rank accuracies of all workers and copy hyperparams and accuracies to history.csv
            best_20perc_workers, worst_20perc_workers, workers_sorted = \
                read_accuracies_and_fill_history_csv(config_train['train_dir'], num_workers, PBT_params,
                                                     PBT_params_worker, config_train['n_epoch'])

        # delete copied ckpts
        ckpt_dir = os.path.join(config_train['train_dir'], 'ckpts')
        for filename in os.listdir(ckpt_dir):
            if filename.startswith('latestcopy'):
                file = os.path.join(ckpt_dir, filename)
                os.remove(file)

        # visualization
        w_params_visualization(config_train['train_dir'], num_workers)

    elif mode == 'test':
        load_ckpt = config_test['load_ckpt']
        train_dir = "/".join(load_ckpt.split("/")[:-2])
        worker_num = ((load_ckpt.split("/")[-1]).split('.')[0]).split('_')[-1]
        config_test_path = os.path.join(train_dir, 'args_test_{}.json'.format(worker_num))
        if os.path.isfile(config_test_path):
            with open(config_test_path, 'rt') as r:
                config_test = json.load(r)
            print('loaded test config from {}'.format(config_test_path))
        config_test['train_dir'] = train_dir
        config_test['load_ckpt'] = load_ckpt
        if 'gpus' in config_test:
            set_cuda_visible_gpus(config_test, 'test')
        Test.test_builder(Namespace(**config_test), )

    else:
        'no correct mode given as input'




if __name__ == '__main__':
    main()