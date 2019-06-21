import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.io import wavfile
from torch.utils import data

warnings.simplefilter("error")


def audio_loader(filename, input_length: int):
    """

    :param filename:
    :param input_length:
    :return:
    """
    _, dt = wavfile.read(filename)
    dt = dt.astype(np.int32)
    amp = np.max(dt) - np.min(dt)
    if len(dt) > input_length:
        max_index = len(dt) - input_length
        offset = np.random.randint(max_index)
        dt = dt[offset: input_length + offset]
    elif len(dt) < input_length:
        total_pad = input_length - len(dt)
        head_pad = np.random.randint(total_pad)
        tail_pad = total_pad - head_pad
        dt = np.pad(dt, (head_pad, tail_pad), 'constant')
    dt = dt.astype(dtype=np.float32) / amp - 0.5
    return torch.from_numpy(dt.copy()).float()


def get_dataset_meta(filename):
    """

    :param filename:
    :return:
    """
    raw_data = pd.read_csv(filename)
    labels = tuple(raw_data.label.unique())
    label_dct = {name: i for i, name in enumerate(labels)}
    return labels, label_dct


def get_test_data(filename):
    """

    :param filename:
    :return:
    """
    dirname = Path(filename).parent
    res_filename = dirname / 'train_validate.csv'
    if res_filename.exists():
        logger.info(f'Reading from {res_filename}')
        return pd.read_csv(res_filename)

    dt = pd.read_csv(filename)
    num_data = len(dt)
    num = num_data // 10
    test_bool = np.array([False] * num_data)
    index_array = np.arange(num_data)
    test_index = np.random.choice(index_array, num)
    test_bool[test_index] = True
    dt['test'] = test_bool
    dt.to_csv(res_filename)
    return dt


class Dataset(data.Dataset):
    """

    """

    def __init__(self,
                 root,
                 data_frame,
                 input_length,
                 num_class,
                 label_dct,
                 data_loader=audio_loader):
        """

        :param root:
        :param data_frame:
        :param input_length:
        :param label_dct:
        :param data_loader:
        """
        self.root = Path(root)
        self.input_length = input_length
        self.__raw_data = data_frame
        self.num_class = num_class
        self.label_dct = label_dct
        self.data_loader = data_loader

    @property
    @lru_cache(maxsize=None)
    def filenames(self):
        return self.__raw_data.fname

    def __len__(self):
        """

        :return:
        """
        return self.__raw_data.shape[0]

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """

        record = self.__raw_data.loc[index]
        name = record.fname
        file_name = self.root / name
        label_name = record.label
        dt = self.data_loader(file_name, self.input_length)
        label = self.label_dct[label_name]
        return dt, torch.tensor(label, dtype=torch.int64)
