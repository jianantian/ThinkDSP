from typing import Union

import torch
from torch import nn
from torch.utils import data


def get_data_statistics(train_dataset: data.Dataset, w: int, h: int):
    """
    calculate mean and std of dataset
    :param train_dataset:
    :param w:
    :param h:
    :return:
    """
    r_sum = torch.tensor(0, dtype=torch.float)
    g_sum = torch.tensor(0, dtype=torch.float)
    b_sum = torch.tensor(0, dtype=torch.float)
    r_square_sum = torch.tensor(0, dtype=torch.float)
    g_square_sum = torch.tensor(0, dtype=torch.float)
    b_square_sum = torch.tensor(0, dtype=torch.float)

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

    r_square_mean = r_square_sum / num
    g_square_mean = g_square_sum / num
    b_square_mean = b_square_sum / num

    r_std = torch.sqrt(r_square_mean - r_mean * r_mean)
    g_std = torch.sqrt(g_square_mean - g_mean * g_mean)
    b_std = torch.sqrt(b_square_mean - b_mean * b_mean)

    data_mean = (r_mean.item(), g_mean.item(), b_mean.item())
    data_std = (r_std.item(), g_std.item(), b_std.item())
    return data_mean, data_std


def validate(net: nn.Module,
             val_data: data.Dataset,
             batch_size: int,
             num_class: int):
    """

    :param net:
    :param val_data:
    :param batch_size:
    :param num_class:
    :return:
    """
    net.eval()

    val_data_iter = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    use_cuda = net.use_cuda

    total_count = torch.zeros(size=(num_class,), dtype=torch.float32)
    true_count = torch.zeros(size=(num_class,), dtype=torch.float32)
    with torch.no_grad():
        for batch_num, (val_data, label) in enumerate(val_data_iter):
            if use_cuda:
                val_data = val_data.cuda()
                label = label.cuda()
            prediction = net(val_data)
            _, pred = torch.max(prediction, 1)
            predict_tensor = (pred == label).squeeze()
            for i, true_label in enumerate(label):
                total_count[true_label] += 1
                true_count[true_label] += predict_tensor[i].item()

    precision = true_count.sum().item() / total_count.sum().item()
    average_precision = true_count / total_count
    mean_ap = average_precision.mean().item()
    res_dct = {'precision': precision, 'map': mean_ap}
    return res_dct, average_precision


def get_parameter_count(model: nn.Module):
    """
    total count of model
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation_layer(activation='relu', inplace=True):
    """

    :param activation:
    :param inplace:
    :return:
    """
    if activation == 'relu':
        activation_layer = nn.ReLU(inplace=inplace)
    elif activation == 'leaky_relu':
        activation_layer = nn.LeakyReLU(0.1, inplace=inplace)
    elif activation == 'sigmoid':
        activation_layer = nn.Sigmoid(inplace=inplace)
    elif activation == 'selu':
        activation_layer = nn.SELU(inplace=inplace)
    elif activation == 'elu':
        activation_layer = nn.ELU(inplace=inplace)
    elif activation == 'tanh':
        activation_layer = nn.Tanh(inplace=inplace)
    else:
        activation_layer = None

    return activation_layer


def get_conv_block(dimension: int,
                   in_channels: int,
                   out_channels: int,
                   kernel_size,
                   stride,
                   padding,
                   activation: Union[str, None] = 'relu',
                   batch_normalize: bool = True,
                   dropout: Union[float, None] = None):
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


def get_conv_block1d(in_channels: int,
                     out_channels: int,
                     kernel_size: int = 3,
                     stride: int = 1,
                     padding: int = 0,
                     activation: Union[str, None] = 'relu',
                     batch_normalize: bool = True,
                     dropout: Union[float, None] = None):
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
                     batch_normalize: bool = True,
                     dropout: Union[float, None] = None):
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


def get_conv_block3d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=1,
                     padding=0,
                     activation: Union[str, None] = 'relu',
                     batch_normalize: bool = True,
                     dropout: Union[float, None] = None):
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
    return get_conv_block(3,
                          in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          padding,
                          activation,
                          batch_normalize,
                          dropout)


def get_linear_block(in_dim, out_dim, activation: Union[str, None] = 'relu', dropout=None):
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


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        kwargs = {'activation': 'relu', 'batch_normalize': False, 'dropout': None}
        self.features = nn.Sequential(
            get_conv_block2d(3, 64, kernel_size=11, stride=4, padding=2, **kwargs),
            self.max_pool,
            get_conv_block2d(64, 192, kernel_size=5, padding=2, **kwargs),
            self.max_pool,
            get_conv_block2d(192, 384, kernel_size=3, padding=1, **kwargs),
            get_conv_block2d(384, 256, kernel_size=3, padding=1, **kwargs),
            get_conv_block2d(256, 256, kernel_size=5, padding=2, **kwargs),
            self.max_pool)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            get_linear_block(256 * 6 * 6, 4096, activation='relu', dropout=0.5),
            get_linear_block(4096, 4096, activation='relu'),
            get_linear_block(4096, num_classes, activation=None))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


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
        self.fc3 = get_linear_block(84, 10, activation=None)

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


if __name__ == '__main__':
    pass
