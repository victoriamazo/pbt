import torch.nn as nn
from collections import OrderedDict


class fc(nn.Module):
    def __init__(self, FLAGS):
        super(fc, self).__init__()
        self.input_dims = FLAGS.width * FLAGS.height
        assert isinstance(self.input_dims, int), 'Please provide int for input_dims'
        self.n_hiddens = list(map(int, FLAGS.n_hiddens.split(',')))
        self.num_classes = FLAGS.num_classes
        self.batch_norm = False
        if hasattr(FLAGS, 'batch_norm'):
            self.batch_norm = FLAGS.batch_norm
        self.keep_prob = 1.0
        if hasattr(FLAGS, 'keep_prob'):
            self.keep_prob = FLAGS.keep_prob

        current_dims = self.input_dims
        layers = OrderedDict()

        for i, n_hidden in enumerate(self.n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            if self.batch_norm:
                layers['bn{}'.format(i + 1)] = nn.BatchNorm1d(n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(1-self.keep_prob)
            current_dims = n_hidden

        layers['out'] = nn.Linear(current_dims, self.num_classes)

        self.model = nn.Sequential(layers)
        # print(self.model)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims

        return self.model.forward(input)