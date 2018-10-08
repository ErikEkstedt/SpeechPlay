import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import WordClassificationDataset


# TODO: Fix nicer plot
def plot_dummy(x, cmap='Blues'):
    plt.figure()
    plt.suptitle('shape: {}x{}'.format( x[0].shape[0], x[0].shape[1]))
    plt.subplot(2,2,1)
    plt.imshow(x[0], cmap=cmap)
    plt.xticks([])
    plt.subplot(2,2,2)
    plt.imshow(x[1], cmap=cmap)
    plt.subplot(2,2,3)
    plt.imshow(x[2], cmap=cmap)
    plt.subplot(2,2,4)
    plt.imshow(x[3], cmap=cmap)
    plt.tight_layout()
    plt.show()


def get_dummy_data(d=10):
    x = np.ones((d,d))
    x_rows = np.ones((d,d))
    x_cols = np.ones((d,d))
    x_rows[::2] = 0
    x_cols[:, ::2] = 0
    x_cross = x_rows + x_cols
    batch = np.stack((x, x_rows, x_cols, x_cross), axis=0)
    return batch


def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5

    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)

    h = int((h_w[0] + (2 * padding[0]) - \
             ( dilation[0] * (kernel_size[0] - 1)) - 1 )/ stride[0] + 1)
    w = int((h_w[1] + (2 * padding[1]) - \
             ( dilation[1] * (kernel_size[1] - 1)) - 1 )/ stride[0] + 1)
    return h, w


class Conv2d(nn.Conv2d):
    '''
    Adding a small function to calculate output shape given input size
    '''
    def get_output_shape(self, h_w):
        h = int((h_w[0] + (2 * self.padding[0]) \
                 - (self.dilation[0] * (self.kernel_size[0] - 1)) \
                 - 1 )/ self.stride[0] + 1)
        w = int((h_w[1] + (2 * self.padding[1]) \
                 - (self.dilation[1] * (self.kernel_size[1] - 1)) \
                 - 1 )/ self.stride[0] + 1)
        return h, w


def plot_spec(s):
    if isinstance(s, torch.Tensor):
        s = s.numpy()
    plt.matshow(s)
    ax = plt.gca()
    plt.gca().invert_yaxis()
    plt.pause(0.1)

# Dataset
dset = WordClassificationDataset(sample=True)

# Convs
kernel_size = (5,5)
stride = (2,2)  # 1 default
padding = 0  # 0 default
dilation = 1  # 1 default
# Regular
conv = Conv2d(in_channels=1,
              out_channels=1,
              kernel_size=kernel_size,
              stride=stride,
              padding=padding,
              dilation=dilation)
conv.weight.data = torch.ones(conv.weight.data.shape) * 0.5
conv.eval()

# data
_, spec, _ = dset[1]
s = torch.tensor(spec.T)
plot_spec(s)
inp = s.unsqueeze(0).unsqueeze(0)  # add channel and batch dimensions
print('input: ', inp.shape)
out = conv(inp)
print('out: ', out.shape)
plot_spec(out.squeeze().detach())





plot_dummy(out.detach().squeeze(1).numpy())




conv = nn.Conv1d(in_channels=1,
              out_channels=1,
              kernel_size=10,
              stride=2,
              padding=padding,
              dilation=dilation)
conv.weight.data = torch.ones(conv.weight.data.shape)
conv.eval()
out = conv(batch)
print('real out shape: ', out.shape)
plot_dummy(out.detach().squeeze(1).numpy())




