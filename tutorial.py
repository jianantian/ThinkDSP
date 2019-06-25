import torch
from torch import nn

from typing import Union


def get_data_statistics(train_dataset, w, h):
    """
    calculate mean and std of dataset
    :param train_dataset:
    :return:
    """
    r_sum, g_sum, b_sum = 0, 0, 0
    r_square_sum, g_square_sum, b_square_sum = 0, 0, 0

    for img, _, in train_dataset:
        r_sum += torch.sum(img[0, :, :])
        g_sum += torch.sum(img[1, :, :])
        b_sum += torch.sum(img[2, :, :])

        r_square_sum += torch.sum(img[0, :, :] ** 2)
        g_square_sum += torch.sum(img[1, :, :] ** 2)
        b_square_sum += torch.sum(img[2, :, :] ** 2)

    num = len(train_dataset) * w * h

    r_mean = r_sum / num
    g_mean = g_sum / num
    b_mean = b_sum / num

    print(g_square_sum)
    r_square_mean = r_square_sum / num
    g_square_mean = g_square_sum / num
    b_square_mean = b_square_sum / num

    r_std = torch.sqrt(r_square_mean - r_mean * r_mean)
    g_std = torch.sqrt(g_square_mean - g_mean * g_mean)
    b_std = torch.sqrt(b_square_mean - b_mean * b_mean)

    data_mean = (r_mean.item(), g_mean.item(), b_mean.item())
    data_std = (r_std.item(), g_std.item(), b_std.item())
    return data_mean, data_std


def get_activation_layer(activation='relu'):
    """

    :param activation:
    :return:
    """
    if activation == 'relu':
        activation_layer = nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        activation_layer = nn.LeakyReLU(0.1, inplace=True)
    elif activation == 'sigmoid':
        activation_layer = nn.Sigmoid(inplace=True)
    elif activation == 'selu':
        activation_layer = nn.SELU(inplace=True)
    elif activation == 'elu':
        activation_layer = nn.ELU(inplace=True)
    elif activation == 'tanh':
        activation_layer = nn.Tanh(inplace=True)
    else:
        activation_layer = None

    return activation_layer


def get_conv_block(dimension,
                   in_channels,
                   out_channels,
                   kernel_size,
                   stride,
                   padding,
                   activation: Union[str, None] = 'relu',
                   batch_normalize=True,
                   dropout=None):
    """

    :param dimension:
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param activation:
    :param batch_normalize:
    :param dropout:
    :return:
    """
    if batch_normalize:
        bias = False
    else:
        bias = True

    model = nn.Sequential()
    if dimension == 1:
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias)
        bn_layer = nn.BatchNorm1d(out_channels)
    elif dimension == 2:
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias)
        bn_layer = nn.BatchNorm2d(out_channels)
    elif dimension == 3:
        conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=bias)
        bn_layer = nn.BatchNorm3d(out_channels)
    else:
        raise ValueError(f'dimension value {dimension} is illegal, should be 1, 2, 3')
    model.add_module('convolution', conv_layer)

    activation_layer = get_activation_layer(activation)

    if activation_layer is not None:
        model.add_module('activation', activation_layer)

    if batch_normalize:
        model.add_module('bn', bn_layer)
    elif dropout is not None:
        model.add_module('dropout', nn.Dropout(dropout))

    return model


def get_conv_block1d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=1,
                     padding=0,
                     activation: Union[str, None] = 'relu',
                     batch_normalize=True,
                     dropout=None):
    """

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param activation:
    :param batch_normalize:
    :param dropout:
    :return:
    """
    return get_conv_block(1,
                          in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          padding,
                          activation,
                          batch_normalize,
                          dropout)


def get_conv_block2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=1,
                     padding=0,
                     activation: Union[str, None] = 'relu',
                     batch_normalize=True,
                     dropout=None):
    """

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param activation:
    :param batch_normalize:
    :param dropout:
    :return:
    """
    return get_conv_block(2,
                          in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          padding,
                          activation,
                          batch_normalize,
                          dropout)


def get_linear_block(in_dim, out_dim, activation='relu', dropout=None):
    """

    :param in_dim:
    :param out_dim:
    :param activation:
    :param dropout:
    :return:
    """
    model = nn.Sequential()
    linear_layer = nn.Linear(in_dim, out_dim)
    model.add_module('linear', linear_layer)

    activation_layer = get_activation_layer(activation)
    if activation_layer is not None:
        model.add_module('activation', activation_layer)

    if dropout is not None:
        model.add_module('dropout', nn.Dropout(dropout))

    return model


class Net(nn.Module):
    """

    """

    def __init__(self, use_cuda=None):
        """

        """
        super().__init__()
        if use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = use_cuda

        self.conv_0 = get_conv_block2d(3, 6, 5, activation='relu')
        self.conv_1 = get_conv_block2d(6, 16, 5, activation='relu')
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = get_linear_block(16 * 5 * 5, 120, activation='relu', dropout=0.5)
        self.fc2 = get_linear_block(120, 84, activation='relu', dropout=0.5)
        self.fc3 = get_linear_block(84, 10, activation='relu')

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.pool(self.conv_0(x))
        x = self.pool(self.conv_1(x))
        x = x.reshape(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def loss(prediction, label):
        """

        :param prediction:
        :param label:
        :return:
        """
        return nn.CrossEntropyLoss()(prediction, label)


class ResidualBlock(nn.Module):
    """
    basic residual block
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, activation='relu'):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param activation:
        """
        super().__init__()

        expansion = out_channels // in_channels
        if expansion == 1:
            downsample = False
        else:
            downsample = True

        out_channels = in_channels * expansion
        self.downsample_layer = None
        if downsample:
            self.downsample_layer = get_conv_block2d(in_channels, out_channels, kernel_size=1, stride=expansion)
        padding = (kernel_size - 1) // 2
        self.conv_0 = get_conv_block2d(in_channels, out_channels, kernel_size, stride=expansion, padding=padding,
                                       activation=activation)
        self.conv_1 = get_conv_block2d(out_channels, out_channels, kernel_size, stride=1, padding=padding,
                                       activation=None)
        self.activation_layer = get_activation_layer(activation)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        residual = x
        x = self.conv_0(x)
        x = self.conv_1(x)

        if self.downsample_layer is not None:
            residual = self.downsample_layer(residual)

        return self.activation_layer(residual + x)


class BottleneckBlock(nn.Module):
    """

    """

    expansion = 4

    def __init__(self, in_channels, out_channels, kernel_size, activation='relu'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        expansion = out_channels // in_channels
        if expansion == self.expansion:
            downsample = False
        else:
            downsample = True

        self.downsample_layer = None
        if downsample:
            self.downsample_layer = get_conv_block2d(in_channels, out_channels, kernel_size=1, stride=2)

        padding = (kernel_size - 1) // 2
        self.conv_0 = get_conv_block2d(in_channels, in_channels, kernel_size=1, stride=1, padding=padding,
                                       activation=activation)
        self.conv_1 = get_conv_block2d(in_channels, in_channels, kernel_size=3, stride=2, padding=padding,
                                       activation=activation)
        self.conv_2 = get_conv_block2d(in_channels, out_channels, kernel_size=1, stride=1, padding=padding,
                                       activation=None)
        self.activation_layer = get_activation_layer(activation)

    def forward(self, x):
        residual = x
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.downsample_layer is not None:
            residual = self.downsample_layer(residual)

        return self.activation_layer(residual + x)


def _get_residual_unit(in_channels, out_channels, num_blocks, kernel_size=3, activation='relu'):

    layers = []
    channels = in_channels
    for i in range(num_blocks):
        block = ResidualBlock(channels, out_channels, kernel_size, activation=activation)
        channels = out_channels
        layers.append(block)
    return nn.Sequential(*layers)


def get_residual_block(in_channels, out_channels, block_number_tuple, kernel_size=3,  activation='relu'):

    layers = []
    channels = in_channels
    for i, num_block in enumerate(block_number_tuple):
        residual_block = _get_residual_unit(channels, out_channels, num_blocks=num_block,
                                            kernel_size=kernel_size, activation=activation)
        channels = out_channels
        layers.append(residual_block)
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """

    """

    def __init__(self, block_number_tuple, num_classes, use_cuda=None):
        """

        """
        super().__init__()

        self.channels = 64
        if use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = use_cuda

        self.num_classes = num_classes
        self.start_conv = get_conv_block2d(3, self.channels, 7, stride=2, padding=3, activation='relu')
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.residual_block = self._get_residual_block(block_number_tuple)

        self.fc = get_linear_block(512, num_classes, activation='relu')
        self.softmax = nn.Softmax(dim=1)

    def _get_residual_block(self, block_number_tuple):
        """

        :return:
        """
        return self._get_residual_block(block_number_tuple)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.start_conv(x)
        x = self.pool(x)
        x = self.residual_block(x)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        return x

    @staticmethod
    def loss(prediction, label):
        """

        :param prediction:
        :param label:
        :return:
        """
        return nn.CrossEntropyLoss()(prediction, label)


if __name__ == '__main__':
    res_net = ResNet(num_classes=10, block_number_tuple=(3, 4, 6, 3))
