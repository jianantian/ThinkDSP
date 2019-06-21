from torch import nn


def get_conv_block(dimension,
                   in_channels,
                   out_channels,
                   kernel_size,
                   stride,
                   padding,
                   activation='relu',
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
    model = nn.Sequential()
    if dimension == 1:
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        bn_layer = nn.BatchNorm1d(out_channels)
    elif dimension == 2:
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        bn_layer = nn.BatchNorm2d(out_channels)
    elif dimension == 3:
        conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        bn_layer = nn.BatchNorm3d(out_channels)
    else:
        raise ValueError(f'dimension value {dimension} is illegal, should be 1, 2, 3')
    model.add_module('convolution', conv_layer)

    if activation == 'relu':
        activation_layer = nn.ReLU()
    elif activation == 'leaky_relu':
        activation_layer = nn.LeakyReLU(0.1)
    elif activation == 'sigmoid':
        activation_layer = nn.Sigmoid()
    elif activation == 'selu':
        activation_layer = nn.SELU()
    elif activation == 'elu':
        activation_layer = nn.ELU()
    elif activation == 'tanh':
        activation_layer = nn.Tanh()
    else:
        activation_layer = None

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
                     padding=1,
                     activation='relu',
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
                     padding=1,
                     activation='relu',
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


class Net(nn.Module):
    """

    """

    def __init__(self):
        """

        """
        super().__init__()
        self.conv_0 = get_conv_block2d(3, 6, 5, activation='relu')
        self.conv_1 = get_conv_block2d(6, 16, 5, activation='relu')
        self.pool = nn.MaxPool2d(kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.pool(self.conv_0(x))
        x = self.pool(self.conv_1(x))
        x = x.reshape(-1, 16 * 5 * 5)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def loss(prediction, label):
        """

        :param prediction:
        :param label:
        :return:
        """
        return nn.CrossEntropyLoss(prediction, label)
