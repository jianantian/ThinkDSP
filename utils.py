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


import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


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
