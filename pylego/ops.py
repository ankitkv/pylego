import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ScaleGradient(Identity):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def backward(self, dx):
        return self.scale * dx


class View(nn.Module):

    def __init__(self, *view_as):
        super().__init__()
        self.view_as = view_as

    def forward(self, x):
        return x.view(*self.view_as)


class Upsample(nn.Module):

    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class GridGaussian(nn.Module):
    '''Projects input coordinates [y, x] to a grid of size [h, w] with a 2D Gaussian of mean [y, x] and std sigma.'''

    def __init__(self, variance, h, w, hmin, hmax, wmin, wmax, mean_value=None):
        super().__init__()
        self.variance = variance
        self.h = h
        self.w = w
        if mean_value is None:
            self.mean_value = 1.0 / (2.0 * np.pi * variance)  # From pdf of Gaussian
        else:
            self.mean_value = mean_value
        ones = np.ones([h, w])
        ys_channel = np.linspace(hmin, hmax, h)[:, np.newaxis] * ones
        xs_channel = np.linspace(wmin, wmax, w)[np.newaxis, :] * ones
        initial = np.concatenate([ys_channel[np.newaxis, ...], xs_channel[np.newaxis, ...]], 0)  # 2 x h x w
        self.linear_grid = nn.Parameter(torch.Tensor(initial), requires_grad=False)

    def forward(self, loc):
        '''loc has shape [..., 2], where loc[...] = [y_i x_i].'''
        loc_grid = loc[..., None, None].expand(*loc.size(), self.h, self.w)
        expanded_lin_grid = self.linear_grid[None, ...].expand_as(loc_grid)
        # both B x 2 x h x w
        reduction_dim = len(loc_grid.size()) - 3
        return ((-(expanded_lin_grid - loc_grid).pow(2).sum(dim=reduction_dim) / (2.0 * self.variance)).exp() *
                self.mean_value)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, rescale=None, norm=None, nonlinearity=F.elu, final=False,
                 skip_last_norm=False, layer_index=1):
        super().__init__()
        self.final = final
        self.skip_last_norm = skip_last_norm
        if stride < 0:
            self.upsample = Upsample(-stride)
            stride = 1
        else:
            self.upsample = Identity()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm is not None:
            self.bn1 = norm(planes, affine=True)
        else:
            self.bn1 = Identity()
        self.nonlinearity = nonlinearity
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        if norm is not None:
            self.bn2 = norm(planes, affine=True)
        else:
            self.bn2 = Identity()
        self.rescale = rescale
        self.stride = stride
        self.gain = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 1, 1)) for _ in range(4)])

        n = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
        self.conv1.weight.data.normal_(0, (layer_index ** (-0.5)) *  np.sqrt(2. / n))
        self.conv2.weight.data.zero_()

    def forward(self, x):
        out = self.upsample(x + self.biases[0])
        out = self.conv1(out) + self.biases[1]
        out = self.bn1(out)
        out = self.nonlinearity(out) + self.biases[2]

        out = self.gain * self.conv2(out) + self.biases[3]
        if not self.final or not self.skip_last_norm:
            out = self.bn2(out)

        if self.rescale is not None:
            x = self.rescale(x)

        out += x
        if not self.final:
            out = self.nonlinearity(out)

        return out


class ResNet(nn.Module):

    def __init__(self, inplanes, layers, block=None, norm=None, nonlinearity=F.elu, skip_last_norm=False,
                 previous_blocks=0):
        '''layers is a list of tuples (layer_size, input_planes, stride). Negative stride for upscaling.'''
        super().__init__()
        self.norm = norm
        self.skip_last_norm = skip_last_norm
        if block is None:
            block = ResBlock

        self.inplanes = inplanes
        self.nonlinearity = nonlinearity
        all_layers = []
        layer_index = 1 + previous_blocks
        for i, (layer_size, inplanes, stride) in enumerate(layers):
            final = (i == len(layers) - 1)
            all_layers.append(self._make_layer(block, inplanes, layer_size, stride=stride, final=final,
                                               layer_index=layer_index))
            layer_index += layer_size
        self.layers = nn.Sequential(*all_layers)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, final=False, layer_index=1):
        rescale = None
        if self.norm is not None:
            batch_norm2d = self.norm(planes * block.expansion, affine=True)
        else:
            batch_norm2d = Identity()
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride < 0:
                stride_ = -stride
                rescale = nn.Sequential(
                    Upsample(stride_),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                    batch_norm2d,
                )
                conv = 1
            else:
                rescale = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    batch_norm2d,
                )
                conv = 0
            n = rescale[conv].kernel_size[0] * rescale[conv].kernel_size[1] * rescale[conv].out_channels
            rescale[conv].weight.data.normal_(0, np.sqrt(2. / n))

        layers = []
        layer_final = final and blocks == 1
        layers.append(block(self.inplanes, planes, stride, rescale, norm=self.norm, nonlinearity=self.nonlinearity,
                            final=layer_final, skip_last_norm=self.skip_last_norm, layer_index=layer_index))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layer_final = final and i == blocks - 1
            layers.append(block(self.inplanes, planes, norm=self.norm, nonlinearity=self.nonlinearity,
                                final=layer_final, skip_last_norm=self.skip_last_norm, layer_index=layer_index+i))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def thresholded_sigmoid(x, linear_range=0.8):
    # t(x)={-l<=x<=l:0.5+x, x<-l:s(x+l)(1-2l), x>l:s(x-l)(1-2l)+2l}
    l = linear_range / 2.0
    return torch.where(x < -l, torch.sigmoid(x + l) * (1. - linear_range),
                       torch.where(x > l, torch.sigmoid(x - l) * (1. - linear_range) + linear_range, x + 0.5))


def inv_thresholded_sigmoid(x, linear_range=0.8):
    # t^-1(x)={0.5-l<=x<=0.5+l:x-0.5, x<0.5-l:-l-ln((1-2l-x)/x), x>0.5+l:l-ln((1-x)/(x-2l))}
    l = linear_range / 2.0
    return torch.where(x < 0.5 - l, -l - torch.log((1. - linear_range - x) / x),
                       torch.where(x > 0.5 + l, l - torch.log((1. - x) / (x - linear_range)), x - 0.5))


def reparameterize_gaussian(mu, logvar, training):
    std = torch.exp(0.5 * logvar)
    if training:
        eps = torch.randn_like(std)
    else:
        eps = torch.zeros_like(std)
    return eps.mul(std).add_(mu)
