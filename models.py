# -*- coding: utf-8 -*-
"""
    IS19.models
    ~~~~~~~~~~~

    Model descriptions

    :copyright: (c) 2019 by Heinrich Dinkel.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from itertools import zip_longest
import pooling


class BiGRU(nn.Module):
    """BiGRU"""

    def __init__(self, inputdim, outputdim, bidirectional=True, **kwargs):
        nn.Module.__init__(self)

        self.rnn = nn.GRU(inputdim, outputdim,
                          bidirectional=bidirectional, batch_first=True, **kwargs)

    def forward(self, x, hid=None):
        x, hid = self.rnn(x)
        return x, (hid,)


class StandardBlock(nn.Module):
    """docstring for StandardBlock"""

    def __init__(self, inputfilter, outputfilter, kernel_size, stride, padding, bn=True, **kwargs):
        super(StandardBlock, self).__init__()
        self.activation = kwargs.get('activation', nn.ReLU(True))
        self.batchnorm = nn.Sequential() if not bn else nn.BatchNorm2d(inputfilter)
        if self.activation.__class__.__name__ == 'GLU':
            outputfilter = outputfilter * 2
        self.conv = nn.Conv2d(inputfilter, outputfilter,
                              kernel_size=kernel_size, stride=stride, bias=not bn, padding=padding)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv(x)
        return self.activation(x)


class ConvConvBlock(nn.Module):
    """docstring for ConvConvBlock"""

    def __init__(self, inputfilter, outputfilter, kernel_size, stride, padding, bn=True, **kwargs):
        super(ConvConvBlock, self).__init__()
        self.conv1 = StandardBlock(
            inputfilter, inputfilter, kernel_size=1, stride=1, bn=True, padding=0, **kwargs)
        self.conv2 = StandardBlock(
            inputfilter, outputfilter, kernel_size=kernel_size, stride=stride, bn=bn, padding=padding, **kwargs)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class CRNN(nn.Module):

    """Encodes the given input into a fixed sized dimension"""

    def __init__(self, inputdim, output_size, **kwargs):
        super(CRNN, self).__init__()
        self._inputdim = inputdim
        self._embed_size = output_size
        self._filtersizes = kwargs.get('filtersizes', [3, 3, 3, 3, 3])
        self._filter = kwargs.get('filter', [16, 32, 128, 128, 128])
        self._pooling = kwargs.get(
            'pooling', [2, 2, (1, 2), (1, 2)])
        self._hidden_size = kwargs.get('hidden_size', 128)
        self._bidirectional = kwargs.get('bidirectional', True)
        self._rnn = kwargs.get('rnn', 'BiGRU')
        self._pooltype = kwargs.get('pooltype', 'MaxPool2d')
        self._activation = kwargs.get('activation', 'ReLU')
        self._blocktype = kwargs.get('blocktype', 'StandardBlock')
        self._bn = kwargs.get('bn', True)
        activkwargs = {}
        if self._activation == 'GLU':
            activkwargs = {'dim': 1}
        poolingtypekwargs = {}
        if self._pooltype == 'LPPool2d':
            poolingtypekwargs = {"norm_type": 4}

        self._filter = [1] + self._filter
        net = nn.ModuleList()
        assert len(self._filter) - 1 == len(self._filtersizes)
        for nl, (h0, h1, filtersize, poolingsize) in enumerate(
                zip_longest(self._filter, self._filter[1:], self._filtersizes, self._pooling)):
            # Stop in zip_longest when last element arrived
            if not h1:
                break
            # For each layer needs to know the filter size
            if self._pooltype in ('ConvolutionPool', 'GatedPooling'):
                poolingtypekwargs = {'filter': h1}
            current_activation = getattr(nn, self._activation)(**activkwargs)
            net.append(globals()[self._blocktype](
                inputfilter=h0, outputfilter=h1, kernel_size=filtersize, padding=int(filtersize)//2, bn=self._bn, stride=1, activation=current_activation))
            # Poolingsize will be None if pooling is finished
            if poolingsize:
                net.append(getattr(pooling, self._pooltype)(
                    kernel_size=poolingsize, **poolingtypekwargs))
            # Only dropout at last layer before GRU
            if nl == (len(self._filter) - 2):
                net.append(nn.Dropout(0.3))
        self.network = nn.Sequential(*net)

        def calculate_cnn_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]
        outputdim = calculate_cnn_size((1, 500, inputdim))
        self.rnn = globals()[self._rnn](
            self._filter[-1] * outputdim[-1], self._hidden_size, self._bidirectional)
        rnn_output = self.rnn(torch.randn(
            1, 500, self._filter[-1]*outputdim[-1]))[0].shape[-1]
        self.outputlayer = nn.Linear(
            rnn_output,
            self._embed_size)
        self.network.apply(self.init_weights)
        self.outputlayer.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x, _ = self.rnn(x)
        x = self.outputlayer(x)
        return x
