from collections import OrderedDict

import torch.nn as nn


class conv(nn.Module):
    def __init__(self, FLAGS):
        super(conv, self).__init__()
        self.num_channels = FLAGS.num_channels
        self.n_filters = list(map(int, FLAGS.n_filters.split(',')))
        self.num_classes = FLAGS.num_classes
        self.batch_norm = False
        if hasattr(FLAGS, 'batch_norm'):
            self.batch_norm = FLAGS.batch_norm
        self.keep_prob = 1.0
        if hasattr(FLAGS, 'keep_prob'):
            self.keep_prob = FLAGS.keep_prob

        current_dims = self.num_channels
        layers = OrderedDict()

        for i, n_filter in enumerate(self.n_filters):
            layers['conv{}'.format(i+1)] = nn.Conv2d(current_dims, n_filter, kernel_size=3, padding=1)
            if self.batch_norm:
                layers['bn{}'.format(i + 1)] = nn.BatchNorm2d(n_filter)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(1-self.keep_prob)
            current_dims = n_filter

        self.model = nn.Sequential(layers)
        current_dims = self.n_filters[-1]*FLAGS.height*FLAGS.width #18432 #self.n_filters[-1] * FLAGS.height/2**len(self.n_filters) * FLAGS.height/2**len(self.n_filters) #18432
        self.classifier = nn.Sequential(nn.Linear(current_dims, self.num_classes))
        # print(self.features)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        x = self.model.forward(input)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

