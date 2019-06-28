from typing import Union, List, Tuple

import torch
from torch import nn
from torch.utils import data
import numpy as np
import cv2


def topk_along_axis(a: np.ndarray,
                    kth: int,
                    axis: int = -1,
                    kind: str = 'introselect',
                    order: Union[str, List[str]] = None,
                    largest: bool = True,
                    sort: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    沿坐标轴 axis 取前 kth 个值, 默认 (largest 为 True) 取最大的 kth 个
    :param a:
    :param kth:
    :param axis:
    :param kind:
    :param order:
    :param largest: True: 最大的 kth 个, False: 最小的 kth 个
    :param sort: 输出是否排序
    :return: 输出 idx, values
    """
    shape = a.shape
    if not largest:
        idx = np.argpartition(a, kth, axis=axis, kind=kind, order=order).take(range(0, kth), axis=axis)
    else:
        idx = np.argpartition(a, -kth, axis=axis, kind=kind, order=order).take(range(shape[axis] - kth, shape[axis]),
                                                                               axis=axis)

    if sort:
        idx = np.take_along_axis(idx,
                                 np.argsort(np.take_along_axis(a, idx, axis=axis), axis=axis),
                                 axis=axis)
        if largest:
            idx = np.flip(idx, axis=axis)
    return idx, np.take_along_axis(a, idx, axis=axis)


def make_idx(idx: np.ndarray, axis: int) -> Tuple[int, ...]:
    """
    由 idx 得到索引 tuple
    :param idx:
    :param axis:
    :return:
    """
    total_idx = np.ogrid[tuple(map(slice, idx.shape))]
    total_idx[axis] = idx
    return tuple(total_idx)


def topk(a: np.ndarray,
         kth: int,
         kind: str = 'introselect',
         order: Union[str, List[str]] = None,
         largest: bool = True,
         sort: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 a 的所有项中取前 kth 项, 默认 (largest 为 True) 取最大的 kth 个
    :param a:
    :param kth:
    :param kind:
    :param order:
    :param largest: True: 最大的 kth 个, False: 最小的 kth 个
    :param sort: 输出是否排序
    :return: 输出 idx, values
    """
    base_a = a.ravel()
    if largest:
        idx = np.argpartition(base_a, a.size - kth, kind=kind, order=order)[-kth:]
    else:
        idx = np.argpartition(base_a, kth, kind=kind, order=order)[:kth]

    if sort:
        idx = idx[np.argsort(base_a[idx])]
        if largest:
            idx = idx[::-1]
    retrieval_idx = np.unravel_index(idx, a.shape)
    idx = np.column_stack(retrieval_idx)
    return idx, a[retrieval_idx]


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


def _resize_img(original_img, input_size, fill_value=0):
    """

    :param original_img:
    :param input_size:
    :param fill_value:
    :return:
    """
    img_h, img_w = original_img.shape[:2]
    w, h = input_size
    scale_rate = min(w / img_w, h / img_h)
    new_w = int(img_w * scale_rate)
    new_h = int(img_h * scale_rate)

    original_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), fill_value=fill_value, dtype=np.uint8)
    h_shift = (h - new_h) // 2
    w_shift = (w - new_w) // 2
    canvas[h_shift: h_shift + new_h, w_shift: w_shift + new_w, :] = original_img
    return canvas


def _rand_scale(scale):
    """

    :param scale:
    :return:
    """
    scale = np.random.uniform(1, scale)
    if np.random.randint(2) < 1:
        return scale
    else:
        return 1. / scale


def random_distort_image(image, hue: int = 15, saturation: float = 1.5, exposure: float = 1.5):
    """
    random change color
    :param image:
    :param hue:
    :param saturation:
    :param exposure:
    :return:
    """
    # determine scale factors
    hue_delta = np.random.uniform(-hue, hue)
    saturation_delta = _rand_scale(saturation)
    exposure_delta = _rand_scale(exposure)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_mat = image[:, :, 0]
    sat_mat = image[:, :, 1]
    exp_mat = image[:, :, 2]

    hue_mat += hue_delta
    hue_mat -= (hue_mat > 180) * 180
    hue_mat += (hue_mat < 0) * 180

    sat_mat *= saturation_delta
    sat_mat[sat_mat > 255.] = 255.

    exp_mat *= exposure_delta
    exp_mat[exp_mat > 255.] = 255.

    img_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img_rgb


def transform_gray(img):
    """

    :param img:
    :return:
    """
    channel = np.random.randint(0, 4)
    if channel > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img[:, :, channel]
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


def flip_image(img: np.ndarray, hflip: bool = False, vflip: bool = False):
    """

    :param img:
    :param hflip:
    :param vflip:
    :return:
    """
    if hflip:
        img = img[:, ::-1, :]
    if vflip:
        img = img[::-1, :, :]
    return img


def rotate_img(img, angle=90):
    """

    :param img:
    :param angle:
    :return:
    """
    h, w, _ = img.shape
    mat = cv2.getRotationMatrix2D((w / 2, h / 2.), angle, 1)
    return cv2.warpAffine(img, mat, (h, w))


def transform_image(original_img, input_size, fill_value=255, new_axis=False, augmentation=True, dtype='float'):
    """
    transform img to torch.tensor of shape NCWH
    :param original_img:
    :param input_size:
    :param fill_value:
    :param new_axis: bool value, default False
    :param augmentation:
    :param dtype: data type to use: float, double, half
    :return:
    """
    img = _resize_img(original_img, input_size, fill_value=fill_value)

    if augmentation:
        # random shift color
        img = random_distort_image(img)

        gray_random = np.random.rand()
        if gray_random > 0.5:
            img = transform_gray(img)

        img = np.array(img / 255., dtype=np.float32)

        hflip_random = np.random.rand()
        vflip_random = np.random.rand()
        hflip = np.all(hflip_random > 0.5)
        vflip = np.all(vflip_random > 0.5)
        img = flip_image(img, hflip=hflip, vflip=vflip)

        rotate_random = np.random.rand()
        rotate_bool = np.all(rotate_random > 0.5)
        if rotate_bool:
            img = rotate_img(img)
        transform_dct = {'hflip': hflip,
                         'vflip': vflip,
                         'rotate': rotate_bool}
    else:
        img = np.array(img / 255., dtype=np.float32)
        transform_dct = {'hflip': False,
                         'vflip': False,
                         'rotate': False}
    img = np.einsum('ijk->kij', img)
    if new_axis:
        img = img[np.newaxis, ...]
    return numpy2tensor(img, dtype=dtype), transform_dct


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
