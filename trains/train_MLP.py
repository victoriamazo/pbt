import os
from tensorboardX import SummaryWriter                                 # pip install tensorboardX
from itertools import chain
import numpy as np

from dataloaders.dataloader_builder import DataLoader
from trains.train_builder import Train
from utils.auxiliary import AverageMeter, save_checkpoint, save_test_losses_to_tensorboard, load_model_and_weights, \
    write_summary_to_csv
import dataloaders.img_transforms as transforms
from metrics.metric_builder import Metric

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()


class train_MPL(Train):
    def __init__(self, FLAGS):
        super(train_MPL, self).__init__(FLAGS)
        self.height = FLAGS.height
        self.width = FLAGS.width
        self.model = FLAGS.model
        if len(FLAGS.decreasing_lr_epochs) > 0:
            self.decreasing_lr_epochs = list(map(int, FLAGS.decreasing_lr_epochs.split(',')))
        else:
            self.decreasing_lr_epochs = None
        self.metric = FLAGS.metric
        self.weight_decay = FLAGS.weight_decay
        self.n_iter = 0
        self.n_epoch = 0
        self.test_iters_dict = {}
        self.results_table_path = os.path.join(self.train_dir, 'results.csv')
        self.loss_summary_path = os.path.join(self.train_dir, 'loss_summary.csv')
        self.debug = FLAGS.debug
        self.load_ckpt = ''
        self.rm_train_dir = True
        if hasattr(FLAGS, 'load_ckpt') and FLAGS.load_ckpt != '':
            self.load_ckpt = FLAGS.load_ckpt
            self.rm_train_dir = False
        if self.worker_num != None:
            self.rm_train_dir = False
            self.writer = None
        else:
            save_path = os.path.join('tensorboard', self.train_dir.split('/')[-1])
            self.writer = SummaryWriter(save_path)

        # seed
        if hasattr(FLAGS, 'seed'):
            self.seed = FLAGS.seed
        torch.manual_seed(self.seed)
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

        self.dataloader, num_samples = DataLoader.dataloader_builder(self.data_loader, self.FLAGS, 'train')
        self.epoch_size = num_samples // self.batch_size


    def _train_one_epoch(self, model, optimizer):
        losses = AverageMeter(precision=4)

        model.train()

        for i, (data, target, _) in enumerate(self.dataloader):
            # transform to pytorch tensor
            totensor = transforms.Compose([transforms.ToTensor(), ])
            data_t = totensor(data)                                 # (batch_size, num_channels, height, width)
            target_t = torch.from_numpy(target)
            target_t = target_t.type(torch.LongTensor)
            target_t = Variable(target_t)
            data_t = Variable(data_t)
            if use_cuda:
                data_t, target_t = data_t.cuda(), target_t.cuda()
            if self.worker_num == None:
                img = ((data[0]-np.min(data[0]))/np.max(data[0]-np.min(data[0]))).astype('float32')
                img = np.stack((img, img, img), axis=2)
                self.writer.add_image('train_images', img, i)

            # run model, get prediction and backprop error
            optimizer.zero_grad()
            output = model(data_t)
            loss = F.cross_entropy(output, target_t)
            losses.update(loss.data[0], self.batch_size)

            # calculate metric and save ckpt
            if self.n_iter % self.num_iters_for_ckpt == 0 and self.n_iter > 0:
                # calculate metric
                logits = output.data.cpu().numpy()
                acc = Metric.metric_builder(self.metric, logits, target, self.FLAGS)
                print('Train: epoch {} (iter {}, {}/{}) Loss: {:.4f} Acc: {:.2f}%'.format(self.n_epoch, self.n_iter, i,
                                                                self.epoch_size, losses.avg[0], acc))

                # add to tensorboard
                if self.worker_num == None:
                    self.writer.add_scalar('loss', losses.avg[0], self.n_iter)
                    self.writer.add_scalar('train_accuracy', acc, self.n_iter)

                # save test losses to tensorboard and results_table.csv
                self.test_iters_dict = save_test_losses_to_tensorboard(self.test_iters_dict, self.results_table_path,
                                                                       self.writer, self.debug)
                # save checkpoint
                states = [{'iteration': self.n_iter, 'epoch': self.n_epoch, 'state_dict': model.module.state_dict()}]
                save_checkpoint(self.ckpts_dir, states, ['model'], worker_num=self.worker_num)

            loss.backward()
            optimizer.step()

            self.n_iter += 1
            if i >= self.epoch_size - 1:
                break

        return losses.avg[0]


    def build(self):
        self._check_args(self.rm_train_dir)

        # initialize or resume training
        _, models, _, self.n_iter, self.n_epoch = load_model_and_weights(self.load_ckpt, self.FLAGS, use_cuda,
                                                                        model=self.model, worker_num=self.worker_num)
        model = models[0]

        # run in parallel on several GPUs
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

        # optimizer
        print('=> setting adam solver')
        parameters = chain(model.parameters())
        optimizer = torch.optim.Adam(parameters, self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)

        # run training for n epochs
        for epoch in range(self.n_epoch, self.num_epochs, 1):
            self.n_epoch = epoch
            if self.n_epoch in self.decreasing_lr_epochs:
                idx = self.decreasing_lr_epochs.index(self.n_epoch) + 1
                self.lr /= 2**idx
                print('learning rate decreases by {} at epoch {}'.format(2**idx, self.n_epoch))

            # run training for one epoch
            train_loss = self._train_one_epoch(model, optimizer)

            # write train and test losses to 'loss_summary.csv
            write_summary_to_csv(self.loss_summary_path, self.results_table_path, self.n_iter, self.n_epoch, train_loss)

































