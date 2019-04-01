import torch
import torch.nn as nn


class MeanMaxPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(MeanMaxPooling, self).__init__()
        from collections import Iterable
        kernel_size = (kernel_size, kernel_size) if not isinstance(
            kernel_size, Iterable) else kernel_size
        assert len(kernel_size) <= 2
        self.meanpool = nn.AvgPool2d((kernel_size))
        self.maxpool = nn.MaxPool2d((kernel_size))

    def forward(self, x):
        return self.meanpool(x) + self.maxpool(x)


class GatedPooling(nn.Module):
    """
        Gated pooling as defined in https://arxiv.org/abs/1509.08985
        This implementation is the LR variant
    """

    def __init__(self, kernel_size, filter):
        super(GatedPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size)
        self.transform = nn.Conv2d(
            filter, filter, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        alpha = torch.sigmoid(self.transform(x))
        return alpha * self.maxpool(x) + (1 - alpha) * self.avgpool(x)


class GatedPooling1(nn.Module):
    """
        Gated pooling as defined in https://arxiv.org/abs/1509.08985
        This implementation is the L variant ( entire layer, one parameter )
    """

    def __init__(self, kernel_size):
        super(GatedPooling1, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size)
        self.transform = nn.Conv2d(
            1, 1, kernel_size=kernel_size, stride=kernel_size)
        torch.nn.init.kaiming_normal_(self.transform.weight)

    def forward(self, x):
        xs = [self.transform(x_filt.unsqueeze(1)).squeeze(1)
              for x_filt in torch.unbind(x, dim=1)]
        alpha = torch.sigmoid(torch.stack(xs, 1))
        return alpha * self.maxpool(x) + (1 - alpha) * self.avgpool(x)


class MixedPooling_fixed_alpha(nn.Module):
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(MixedPooling_fixed_alpha, self).__init__()

        from collections import Iterable
        kernel_size = (kernel_size, kernel_size) if not isinstance(
            kernel_size, Iterable) else kernel_size
        self.alpha = kwargs.get('alpha', 0.3)
        assert len(kernel_size) <= 2
        self.meanpool = nn.AvgPool2d((kernel_size))
        self.maxpool = nn.MaxPool2d((kernel_size))

    def forward(self, x):
        return (1 - self.alpha)*self.meanpool(x) + (self.alpha * self.maxpool(x))


class MixedPooling_alpha_beta(nn.Module):
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(MixedPooling_alpha_beta, self).__init__()

        from collections import Iterable
        kernel_size = (kernel_size, kernel_size) if not isinstance(
            kernel_size, Iterable) else kernel_size
        alpha_val = kwargs.get('alpha', 0.3)
        beta_val = kwargs.get('beta', 0.3)
        self.beta = nn.Parameter(torch.tensor(beta_val))
        self.alpha = nn.Parameter(torch.tensor(alpha_val))
        assert len(kernel_size) <= 2
        self.meanpool = nn.AvgPool2d((kernel_size))
        self.maxpool = nn.MaxPool2d((kernel_size))

    def forward(self, x):
        return self.beta*self.meanpool(x) + self.alpha * self.maxpool(x)


class MixedPooling_learn_alpha(nn.Module):
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(MixedPooling_learn_alpha, self).__init__()

        from collections import Iterable
        kernel_size = (kernel_size, kernel_size) if not isinstance(
            kernel_size, Iterable) else kernel_size
        alpha_val = kwargs.get('alpha', 0.3)
        self.alpha = nn.Parameter(torch.tensor(alpha_val))
        assert len(kernel_size) <= 2
        self.meanpool = nn.AvgPool2d((kernel_size))
        self.maxpool = nn.MaxPool2d((kernel_size))

    def forward(self, x):
        return ((1.-self.alpha)*self.meanpool(x)) + (self.alpha * self.maxpool(x))


class ConvolutionPool(nn.Module):

    """Docstring for ConvolutionPool. """

    def __init__(self, kernel_size, filter, stride=None, **kwargs):
        """Inits the class

        :kernel_size: TODO
        :stride: TODO

        """
        nn.Module.__init__(self)

        self._kernel_size = kernel_size
        self.conv = nn.Conv2d(
            filter, filter, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, x):
        return self.conv(x)


LPPool2d = nn.LPPool2d
MaxPool2d = nn.MaxPool2d
AvgPool2d = nn.AvgPool2d
