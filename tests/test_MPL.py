import itertools
import os
import numpy as np

import dataloaders.img_transforms as transforms
from dataloaders.dataloader_builder import DataLoader
from metrics.metric_builder import Metric
from tests.test_builder import Test
from utils.auxiliary import AverageMeter, save_loss_to_resultstable, check_if_best_model_and_save, \
    load_model_and_weights

import torch
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()



class test_MPL(Test):
    def __init__(self, FLAGS):
        super(test_MPL, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.metric = FLAGS.metric
        self.model = FLAGS.model
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        if self.worker_num != None:
            self.results_table_path = os.path.join(self.train_dir, 'results_{}.csv'.format(self.worker_num))
        else:
            self.results_table_path = os.path.join(self.train_dir, 'results.csv')

        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        self.dataloader, num_samples = DataLoader.dataloader_builder(self.data_loader, self.FLAGS, 'test')
        self.epoch_size = num_samples // self.batch_size


    def _test(self, model):
        losses = AverageMeter(precision=4)
        x, y = [], []

        model.eval()

        for i, (data, target, _) in enumerate(self.dataloader):
            # transform to pytorch tensor
            totensor = transforms.Compose([transforms.ToTensor(), ])
            data_t = totensor(data)  # (batch_size, num_channels, height, width)
            target_t = torch.from_numpy(target)
            target_t = target_t.type(torch.LongTensor)
            target_t = Variable(target_t)
            data_t = Variable(data_t)
            if use_cuda:
                data_t, target_t = data_t.cuda(), target_t.cuda()

            # run test and get predictions
            output = model(data_t)
            loss = F.cross_entropy(output, target_t)
            losses.update(loss.data.item(), self.batch_size)
            x.append(output.data.cpu().numpy()[:])
            y.append(target[:])

            if i >= self.epoch_size - 1:
                break

        # calculate accuracy
        logits = np.array(list(itertools.chain(*x)))
        y = np.array(list(itertools.chain(*y)))
        test_acc = Metric.metric_builder(self.metric, logits, y, self.FLAGS)
        print('\t\t\tTest (epoch {}, iter {}): Average loss: {:.4f}, Accuracy: {:.2f}%'.format(self.n_epoch, self.n_iter,
                                                                                               losses.avg[0], test_acc))
        return losses.avg[0], test_acc


    def build(self):
        self._check_args()

        # load models and weights
        models_loaded, models, model_names, self.n_iter, self.n_epoch = load_model_and_weights(self.load_ckpt,
                                    self.FLAGS, use_cuda, model=self.model, ckpts_dir=self.ckpts_dir, train=False,
                                    worker_num=self.worker_num)
        model = models[0]

        if models_loaded:
            # run test
            test_loss, test_acc = self._test(model)

            # save test losses and metrics to results table
            col_names = ['test_loss', 'test_acc']
            values = [test_loss, test_acc]
            save_loss_to_resultstable(values, col_names, self.results_table_path, self.n_iter, self.n_epoch, self.debug)

            # check if best model (saves best model not in debug mode)
            save_path = os.path.join(self.train_dir, 'ckpts')
            check_if_best_model_and_save(self.results_table_path, models, model_names, self.n_iter,
                                         self.n_epoch, save_path, self.debug, self.worker_num)
        else:
            print('no ckpt found for running test')
































