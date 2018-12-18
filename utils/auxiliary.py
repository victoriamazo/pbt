import csv
import os
import shutil
from collections import OrderedDict
import numpy as np
import pandas as pd
from path import Path
from shutil import copyfile

from models.model_builder import Model

import torch


def idx2label(data_dir):
    dataset = data_dir.split('/')[-1]
    label_dict = {}
    if dataset == 'mnist':
        label_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '0': 0}
    return label_dict


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0
        self.meters = i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)

    def __len__(self):
        return self.meters


def check_if_best_model_and_save(results_table_path, models, model_names, iter, epoch, save_path, debug,
                                 worker_num=None, best_criteria='test_acc'):
    ''' Get test loss and metric from the results_table,
        decide whether the current metric is the best.
        The decision is made based according to 'best_criteria', which should be a
        name of column in the results table.
        If best, save the ckpt as best (not in debug mode).'''
    filename = (results_table_path.split('/')[-1]).split('.')[0]
    results_table_path_tmp = Path(results_table_path).dirname() / '{}_tmp.csv'.format(filename)

    # check whether 'iter' is the best iteration according to the best criteria
    is_best = False
    if not debug:
        if os.path.isfile(results_table_path):
            copyfile(results_table_path, results_table_path_tmp)
            results_table = pd.read_csv(results_table_path_tmp, index_col=0)
            if len(results_table.index) > 0:
                col_names = results_table.columns.tolist()[1:]
                assert best_criteria in col_names, 'criteria for best model is not in the results table'
                best_criteria_col_np = np.array(results_table[best_criteria])
                best_criteria_value = np.max(best_criteria_col_np)
                iter_col_np = np.array(results_table['iter'])
                assert iter in iter_col_np, 'current iteration is not in the results table'
                iter_idx = np.where(np.array(results_table['iter'])==iter)[0][0]
                iter_criteria_value = best_criteria_col_np[iter_idx]
                if iter_criteria_value >= best_criteria_value:
                    is_best = True
            os.remove(results_table_path_tmp)

        # save best model
        if is_best:
            states = []
            for model in models:
                states.append({'iteration': iter, 'epoch': epoch, 'state_dict': model.state_dict()})
            save_checkpoint(save_path, states, model_names, is_best=True, worker_num=worker_num)

    return is_best


def save_checkpoint(save_path, states, state_names, worker_num=None, is_best=False, filename='.pth.tar'):
    if is_best:
        for (prefix, state) in zip(state_names, states):
            if worker_num != None:
                best_ckpt_path = os.path.join(save_path, '{}_ckpt_best_{}.pth.tar'.format(prefix, worker_num))
            else:
                best_ckpt_path = os.path.join(save_path, '{}_ckpt_best.pth.tar'.format(prefix))
            torch.save(state, best_ckpt_path)
            print('saved best {} model (iter {}) to {}'.format(prefix, state['iteration'], save_path))
    else:
        for (prefix, state) in zip(state_names, states):
            if worker_num != None:
                ckpt_path = os.path.join(save_path, '{}_ckpt_{}{}'.format(prefix, worker_num, filename))
            else:
                ckpt_path = os.path.join(save_path, '{}_ckpt{}'.format(prefix, filename))
            torch.save(state, ckpt_path)


def load_model_and_weights(load_ckpt, FLAGS, use_cuda, model='conv', worker_num=None, ckpts_dir=None, train=True):
    # load models
    model_net = Model.model_builder(model, FLAGS)
    if use_cuda:
        model_net = model_net.cuda()
    models = [model_net]
    model_names = ['model']
    model_loaded = False

    # load weights
    model_path = ''
    if load_ckpt != '':
        model_path = load_ckpt
    elif load_ckpt == '' and not train:
        if worker_num != None:
            model_path = os.path.join(ckpts_dir, 'model_ckpt_{}.pth.tar'.format(worker_num))
        else:
            model_path = os.path.join(ckpts_dir, 'model_ckpt.pth.tar')
    n_iter, n_epoch, train_iter = 0, 0, 0
    if os.path.isfile(model_path):
        model_path = Path(model_path)
        ckpt_dir = model_path.dirname()
        if worker_num != None:
            model_path_tmp = ckpt_dir / 'model_ckpt_tmp_{}.pth.tar'.format(worker_num)
        else:
            model_path_tmp = ckpt_dir / 'model_ckpt_tmp.pth.tar'
        copyfile(model_path, model_path_tmp)
        model_ckpt_dict = torch.load(model_path_tmp)
        if 'iteration' in model_ckpt_dict:
            train_iter = model_ckpt_dict['iteration']
        if 'epoch' in model_ckpt_dict:
            epoch = model_ckpt_dict['epoch']
        assert isinstance(model_ckpt_dict['state_dict'], (dict, OrderedDict)), type(model_ckpt_dict['state_dict'])
        model_net.load_state_dict(model_ckpt_dict['state_dict'], strict=False)
        model_loaded = True
        os.remove(model_path_tmp)

        if train:
            print('Model training resumed from the ckpt {} (epoch {}, iter {})'.format(model_path, epoch, train_iter))
        else:
            print('Model ckpt loaded from {} (epoch {}, iter {})'.format(model_path, epoch, train_iter))

        if train_iter > 0:
            if train:
                n_iter = train_iter + 1
            else:
                n_iter = train_iter
            n_epoch = epoch
    else:
        if train:
            model_net.init_weights()

    return model_loaded, models, model_names, n_iter, n_epoch


def save_test_losses_to_tensorboard(test_iters_dict, results_table_path, writer, debug=False):
    ''' Get test loss from the results_table and add them to tensorboard'''
    filename = (results_table_path.split('/')[-1]).split('.')[0]
    results_table_path_tmp = Path(results_table_path).dirname() / '{}_tmp.csv'.format(filename)
    if os.path.isfile(results_table_path):
        copyfile(results_table_path, results_table_path_tmp)
        results_table = pd.read_csv(results_table_path_tmp, index_col=0)
        if len(results_table.index) > 0:
            col_names = results_table.columns.tolist()[2:]
            for row_idx in range(len(results_table.index)):
                test_iter = int(results_table.iloc[row_idx]['iter'])
                if test_iter not in test_iters_dict:
                    test_iters_dict[test_iter] = 1
                    if not debug and writer is not None:
                        for col_name in col_names:
                            writer.add_scalar(col_name+'_test',  results_table.iloc[row_idx][col_name], test_iter)

        if os.path.isfile(results_table_path_tmp):
            os.remove(results_table_path_tmp)

    return test_iters_dict


def save_loss_to_resultstable(values_list, col_names, results_table_path, n_iter, epoch, debug=False):
    '''Saves test losses and metrics to results table (not in debug mode)'''
    assert len(values_list) == len(col_names)

    if not debug:
        # init results table
        row_idx = 0
        if not os.path.isfile(results_table_path):
            columns = ['epoch'] + ['iter'] + col_names
            index = np.arange(1)
            results_table = pd.DataFrame(columns=columns, index=index)
        else:
            results_table = pd.read_csv(results_table_path, index_col=0)
            if len(results_table.index) > 0:
                iters = list(results_table['iter'])
                if float(n_iter) in iters:
                    row_idx = iters.index(n_iter)
                else:
                    row_idx = len(results_table.index)

        # write all values to the results table
        results_table.ix[row_idx, 'epoch'] = epoch
        results_table.ix[row_idx, 'iter'] = n_iter
        for i, value in enumerate(values_list):
            results_table.ix[row_idx, col_names[i]] = value
        results_table.to_csv(results_table_path, index=True)


def write_summary_to_csv(loss_summary_path, results_table_path, n_iter, epoch, train_loss):
    ''' define results and loss writers '''

    if not os.path.isfile(loss_summary_path):
        with open(loss_summary_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['epoch', 'iter', 'train_loss', 'test_loss'])

    # get test loss for current iteration
    test_loss = -1
    filename = (results_table_path.split('/')[-1]).split('.')[0]
    results_table_path_tmp = Path(results_table_path).dirname() / '{}_tmp.csv'.format(filename)
    if os.path.isfile(results_table_path):
        copyfile(results_table_path, results_table_path_tmp)
        results_table = pd.read_csv(results_table_path_tmp, index_col=0)
        if len(results_table.index) > 0:
            iter_col_np = np.array(results_table['iter'])
            if n_iter in iter_col_np:
                iter_idx = np.where(np.array(results_table['iter']) == n_iter)[0][0]
                test_loss = results_table.iloc[iter_idx]['test_loss']
        os.remove(results_table_path_tmp)

    # write train and test losses to 'loss_summary.csv
    with open(loss_summary_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([epoch, n_iter, train_loss, test_loss])

    return test_loss



























