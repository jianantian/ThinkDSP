from typing import Union, Tuple

import torch
from loguru import logger
from torch import nn

from base import get_conv_block2d, get_linear_block, get_activation_layer


class ResidualBlock(nn.Module):
    """
    basic residual block
    """
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 activation: Union[str, None] = 'relu',
                 **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param stride:
        :param kernel_size:
        :param activation:
        :param kwargs:
        """
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.downsample_layer = None
        if in_channels != out_channels * self.expansion or stride != 1:
            self.downsample_layer = get_conv_block2d(in_channels,
                                                     out_channels,
                                                     kernel_size,
                                                     stride=stride,
                                                     padding=padding)
        self.conv_0 = get_conv_block2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       activation=activation)
        self.conv_1 = get_conv_block2d(out_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=1,
                                       padding=padding,
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

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 activation: Union[str, None] = 'relu',
                 downsample_avg_pool: bool = False):

        super().__init__()

        self.downsample_layer = None
        if in_channels != out_channels * self.expansion or stride != 1:
            if not downsample_avg_pool:
                self.downsample_layer = get_conv_block2d(in_channels,
                                                         out_channels * self.expansion,
                                                         kernel_size=1,
                                                         stride=stride)
            else:
                self.downsample_layer = nn.Sequential(
                    nn.AvgPool2d(kernel_size=(2, 2)),
                    get_conv_block2d(in_channels,
                                     out_channels * self.expansion,
                                     kernel_size=1,
                                     stride=1)
                )
        padding = (kernel_size - 1) // 2
        self.conv_0 = get_conv_block2d(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       activation=activation)
        self.conv_1 = get_conv_block2d(out_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       activation=activation)
        self.conv_2 = get_conv_block2d(out_channels,
                                       out_channels * self.expansion,
                                       kernel_size=1,
                                       stride=1,
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


def get_residual_unit(block: Union[ResidualBlock, BottleneckBlock],
                      in_channels: int,
                      out_channels: int,
                      num_blocks: int,
                      stride: int = 1,
                      kernel_size: Union[int, Tuple[int, int]] = 3,
                      activation: Union[str, None] = 'relu',
                      downsample_avg_pool: bool = False):
    """

    :param block:
    :param in_channels:
    :param out_channels:
    :param num_blocks:
    :param stride:
    :param kernel_size:
    :param activation:
    :param downsample_avg_pool:
    :return:
    """
    layers = []
    channels = in_channels
    stride = stride
    for i in range(num_blocks):
        logger.debug(f'in_channels: {channels}, out_channels: {out_channels}, stride: {stride}')
        conv_layer = block(channels,
                           out_channels,
                           stride,
                           kernel_size,
                           activation=activation,
                           downsample_avg_pool=downsample_avg_pool)
        channels = out_channels * block.expansion
        stride = 1
        layers.append(conv_layer)
    return nn.Sequential(*layers)


def get_residual_module(block: Union[ResidualBlock, BottleneckBlock],
                        in_channels: int,
                        block_number_tuple: Tuple[int, ...],
                        kernel_size: int = 3,
                        activation: Union[str, None] = 'relu',
                        downsample_avg_pool: bool = False):
    """

    :param block:
    :param in_channels:
    :param block_number_tuple:
    :param kernel_size:
    :param activation:
    :param downsample_avg_pool:
    :return:
    """
    layers = []
    channels = in_channels
    out_channels = channels
    stride = 1
    for num_block in block_number_tuple:
        residual_block = get_residual_unit(block,
                                           channels,
                                           out_channels,
                                           num_blocks=num_block,
                                           stride=stride,
                                           kernel_size=kernel_size,
                                           activation=activation,
                                           downsample_avg_pool=downsample_avg_pool)
        channels = out_channels * block.expansion
        out_channels = out_channels * 2
        stride = 2
        layers.append(residual_block)
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """

    """

    def __init__(self,
                 block: Union[ResidualBlock, BottleneckBlock],
                 block_number_tuple: Tuple[int, ...],
                 num_classes: int,
                 basic_channel: int,
                 start_block: Union[nn.Module, None] = None,
                 use_small_kernel: bool = False,
                 downsample_avg_pool: bool = False,
                 use_cuda: Union[bool, None] = None):
        """

        """
        super().__init__()

        self.block = block
        self.basic_channel = basic_channel
        if use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = use_cuda

        self.num_classes = num_classes

        if start_block is not None:
            self.start_block = start_block
        elif not use_small_kernel:
            self.start_block = nn.Sequential(
                get_conv_block2d(3, self.basic_channel, 7, stride=2, padding=3, activation='relu'),
                nn.MaxPool2d(kernel_size=2)
            )
        else:
            # replace 7 * 7 conv kernel with three 3 * 3 kernel
            self.start_block = nn.Sequential(
                get_conv_block2d(3, 32, kernel_size=3, stride=2, padding=1, activation='relu'),
                get_conv_block2d(32, 32, kernel_size=3, stride=1, padding=1, activation='relu'),
                get_conv_block2d(32, 64, kernel_size=3, stride=1, padding=1, activation='relu'),
                nn.MaxPool2d(kernel_size=2)
            )

        self.residual_block = get_residual_module(self.block,
                                                  self.basic_channel,
                                                  block_number_tuple,
                                                  downsample_avg_pool=downsample_avg_pool)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        channel_expand = 2 ** (len(block_number_tuple) - 1)
        self.fc = get_linear_block(self.basic_channel * channel_expand * block.expansion,
                                   num_classes,
                                   activation=None)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.start_block(x)
        x = self.residual_block(x)
        x = self.avg_pool(x)
        x = x.reshape(x.size(0), -1)
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


class ResNetImagenet(ResNet):
    """
    resnet for imagenet
    """

    def __init__(self,
                 block: Union[ResidualBlock, BottleneckBlock],
                 block_number_tuple: Tuple[int, ...],
                 use_small_kernel: bool = False,
                 downsample_avg_pool: bool = False,
                 use_cuda: Union[bool, None] = None):
        """

        :param block:
        :param block_number_tuple:
        :param use_small_kernel:
        :param downsample_avg_pool:
        :param use_cuda:
        """
        super().__init__(block,
                         block_number_tuple,
                         num_classes=1000,
                         basic_channel=64,
                         start_block=None,
                         use_small_kernel=use_small_kernel,
                         downsample_avg_pool=downsample_avg_pool,
                         use_cuda=use_cuda)


class ResNetCifar(ResNet):
    """
    resnet for cifar10
    """

    def __init__(self,
                 block_number_tuple: Tuple[int, ...],
                 use_cuda: Union[bool, None] = None):
        super().__init__(ResidualBlock,
                         block_number_tuple,
                         num_classes=10,
                         basic_channel=16,
                         start_block=get_conv_block2d(3, 16, kernel_size=3, stride=1, padding=1),
                         use_small_kernel=False,
                         downsample_avg_pool=False,
                         use_cuda=use_cuda)


class ResNet18(ResNetImagenet):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block=ResidualBlock, block_number_tuple=(2, 2, 2, 2), use_cuda=use_cuda)


class ResNet34(ResNetImagenet):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block=ResidualBlock, block_number_tuple=(3, 4, 6, 3), use_cuda=use_cuda)


class ResNet50(ResNetImagenet):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block=BottleneckBlock, block_number_tuple=(3, 4, 6, 3), use_cuda=use_cuda)


class ResNet101(ResNetImagenet):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block=BottleneckBlock, block_number_tuple=(3, 4, 23, 3), use_cuda=use_cuda)


class ResNet152(ResNetImagenet):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block=BottleneckBlock, block_number_tuple=(3, 8, 36, 3), use_cuda=use_cuda)


class ResNet20(ResNetCifar):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block_number_tuple=(3, 3, 3), use_cuda=use_cuda)


class ResNet32(ResNetCifar):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block_number_tuple=(5, 5, 5), use_cuda=use_cuda)


class ResNet44(ResNetCifar):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block_number_tuple=(7, 7, 7), use_cuda=use_cuda)


class ResNet56(ResNetCifar):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block_number_tuple=(9, 9, 9), use_cuda=use_cuda)


class ResNet110(ResNetCifar):
    """

    """

    def __init__(self, use_cuda: Union[bool, None] = None):
        super().__init__(block_number_tuple=(18, 18, 18), use_cuda=use_cuda)
