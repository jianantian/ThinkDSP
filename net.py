import json
import time
from collections import deque
from pathlib import Path

import torch
from loguru import logger
from torch import nn, optim
from torch.utils import data

import utils
import visualize

logger.add('train_audionet.log', level='INFO', colorize=True, format="<green>{time}</green> <level>{message}</level>")


class PrevLoss:
    def __init__(self):
        self.__container = deque([])

    def __len__(self):
        return len(self.__container)

    def pop(self):
        """
        """
        return self.__container.popleft()

    def append(self, x):
        """
        """
        container = self.__container
        while len(container) >= 3:
            container.pop()
        container.append(x)
        self.__container = container

    def value(self):
        return min(self.__container)


class Config(object):
    """

    """
    __slots__ = ('root', 'train_fname', 'epoch', 'batch_size', 'input_length',
                 'save_interval', 'val_interval', 'log_interval', 'lr', 'lr_decay',
                 'lr_decay_period', 'lr_decay_epoch', 'wd')

    def __init__(self, dct):
        """

        :param dct:
        """
        for k in self.__slots__:
            v = dct.get(k, None)
            if k in {"train_root", "val_root", "trainval_root"}:
                v = Path(v)
            setattr(self, k, v)


def parse_config(config_path):
    """

    :param config_path:
    :return:
    """
    with open(config_path, 'r') as fr:
        dct = json.load(fr)
    return Config(dct)


class NamedLayer(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.name = name


def get_subsample_layer(out_channels, kernel_size):
    model = NamedLayer(f'subsample')
    layer = nn.Sequential(nn.Conv1d(1, out_channels, kernel_size=kernel_size, stride=kernel_size),
                          nn.LeakyReLU(0.1),
                          nn.BatchNorm1d(out_channels))
    model.add_module('subsample', layer)
    return model


def get_block(index, in_channels, out_channels, kernel_size):
    """
    """
    model = NamedLayer(f'convolutional_{index}')
    layer = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
                          nn.LeakyReLU(0.1),
                          nn.BatchNorm1d(out_channels),
                          nn.MaxPool1d(kernel_size))
    model.add_module(f'convolutional', layer)
    return model


def get_softmax_layer(in_channels, num_class):
    """
    """
    model = NamedLayer('softmax')
    linear_layer = nn.Linear(in_channels, num_class)
    model.add_module(f'linear', linear_layer)
    softmax_layer = nn.Softmax(dim=1)
    model.add_module(f'softmax', softmax_layer)
    return model


def validate(net, val_data, batch_size):
    net.eval()

    val_data_iter = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    use_cuda = net.use_cuda

    total_num = 0
    true_detect_num = 0
    with torch.no_grad():
        for i, (val_data, label) in enumerate(val_data_iter):
            if use_cuda:
                val_data = val_data.cuda()
                label = label.cuda()
            prediction = net(val_data)
            _, prediction_label = torch.max(prediction, dim=1)

            total_num += label.shape[0]
            true_detect_num += (prediction_label == label).sum().item()

    val_mean_precision = true_detect_num / total_num

    validation_res_dct = {'precision': val_mean_precision}
    return validation_res_dct


class AudioNet(nn.Module):
    """
    """

    def __init__(self,
                 input_length,
                 name_tuple,
                 use_cuda=torch.cuda.is_available()):
        super().__init__()
        self.input_length = input_length
        self.name_tuple = name_tuple
        self.num_class = len(name_tuple)
        self.use_cuda = use_cuda
        self._init()

    def _init(self):
        subsample_block = get_subsample_layer(128, kernel_size=3)
        #         self.add_module('subsample_block', subsample_block)
        self.subsample_layer = subsample_block

        conv_block = NamedLayer('convolution')

        base_index = 0
        for i in range(2):
            sub_conv_block = get_block(i + base_index, in_channels=128, out_channels=128, kernel_size=3)
            conv_block.add_module(f'conv_block_{i + base_index}', sub_conv_block)

        base_index += 2

        sub_conv_block = get_block(base_index, in_channels=128, out_channels=256, kernel_size=3)
        conv_block.add_module('conv_block_2', sub_conv_block)
        base_index += 1

        for i in range(5):
            sub_conv_block = get_block(i + base_index, in_channels=256, out_channels=256, kernel_size=3)
            conv_block.add_module(f'conv_block_{i + base_index}', sub_conv_block)
        base_index += 5

        sub_conv_block = get_block(base_index, in_channels=256, out_channels=512, kernel_size=3)
        conv_block.add_module(f'conv_block_{base_index}', sub_conv_block)

        end_conv_block = nn.Sequential(nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.1),
                                       nn.Dropout(0.5))
        conv_block.add_module(f'end_conv_block', end_conv_block)

        # self.add_module('convolution', conv_block)
        self.convolution_layer = conv_block

        softmax_block = get_softmax_layer(512, self.num_class)
        # self.add_module('softmax_block', softmax_block)
        self.softmax_layer = softmax_block

    def forward(self, x):
        """
        """
        x = torch.unsqueeze(x, dim=1)
        x = self.subsample_layer(x)
        x = self.convolution_layer(x)

        b, c, l = x.shape
        size = c * l
        x = x.reshape((b, size,))

        x = self.softmax_layer(x)
        return x

    def loss(self, prediction, target):
        """
        """
        loss = nn.CrossEntropyLoss()
        return loss(prediction, target)


if __name__ == '__main__':
    root = Path('/data/FSDKaggle2018/FSDKaggle2018.audio_train')
    train_fname = Path('/data/FSDKaggle2018/FSDKaggle2018.meta/train_post_competition.csv')
    labels, label_dct = utils.get_dataset_meta(train_fname)

    num_class = len(labels)
    full_df = utils.get_test_data(train_fname)

    train_df = full_df[full_df.test == False]
    train_df.index = range(len(train_df))

    test_df = full_df[full_df.test == True]
    test_df.index = range(len(test_df))

    config = parse_config('config.json')
    input_length = config.input_length
    train_dataset = utils.Dataset(root, train_df, input_length, num_class=num_class, label_dct=label_dct)
    train_data_iter = data.DataLoader(train_dataset, batch_size=8)

    net = AudioNet(input_length=input_length, name_tuple=labels)

    learning_rate = config.lr
    weight_decay = config.wd

    val_interval = config.val_interval
    save_interval = config.save_interval
    log_interval = config.log_interval

    lr_decay_epoch = set(config.lr_decay_epoch)
    lr_decay = config.lr_decay

    epoch = config.epoch
    batch_size = config.batch_size

    use_cuda = net.use_cuda
    if use_cuda:
        net = net.cuda()

    learning_rate = 0.01

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    visualizer = visualize.Visualizer('AudioNet')

    prev_loss_list = PrevLoss()
    prev_loss = float('inf')
    for e in range(epoch):
        need_save = False
        start_time = time.time()
        net.train()
        train_dataset = utils.Dataset(root, train_df, input_length, num_class=num_class, label_dct=label_dct)
        test_dataset = utils.Dataset(root, test_df, input_length, num_class=num_class, label_dct=label_dct)

        train_data_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        running_loss = 0.0
        for i, (train_data, label) in enumerate(train_data_iter):
            if use_cuda:
                train_data = train_data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            prediction = net(train_data)
            loss = net.loss(prediction, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if e > 10:
            prev_loss_list.append(running_loss)
            prev_loss = prev_loss_list.value()

        if prev_loss < running_loss - 0.01:
            learning_rate = learning_rate / 2
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        valid_train_res = validate(net, train_dataset, batch_size)
        valid_test_res = validate(net, test_dataset, batch_size)
        loss_dct = {'loss': running_loss}
        val_precision = valid_train_res['precision']
        test_precision = valid_test_res['precision']
        logger.info(
            f'epoch num: {e}, loss: {running_loss}, val_precision: {val_precision}, test_precision: {test_precision}, '
            f'learning_rate: {learning_rate}, time: {time.time() - start_time}')
        visualizer.plot(loss_dct)
        visualizer.plot(valid_train_res)
        visualizer.plot(valid_test_res)
