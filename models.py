import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import os
from os.path import join

class CheckpointSaver(object):
    def __init__(self, root='checkpoints', run=None, model_name='CNN_speech'):
        if not os.path.exists(root):
            os.makedirs(root) 

        if run is None:
            now = datetime.datetime.now()
            save_dir = os.path.join(root, now.strftime("%Y-%m-%d_%H:%M"))
            os.makedirs(save_dir)
        else:
            save_dir = os.path.join(root, run)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        self.save_dir = save_dir
        self.model_name = model_name

    def save(self, model, optimizer, epoch, loss):
        filename =  '{}_ep{}_loss{}.tar'.format(self.model_name, epoch, loss)
        filename = join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'loss': loss,
        }, filename)

# TODO
# Use transfer learning for early layers
#  - resnet base
#  - vgg base
#  - vgg base

class CNN(nn.Module):
    def __init__(self, input_dims=(99, 161), out_classes=30, hidden_channels=128):
        super(CNN, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=1,
                               out_channels=hidden_channels,
                               kernel_size=5,
                               stride=2)
        self.conv1 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=5,
                               stride=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=5,
                               stride=2)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=5,
                               stride=2)

        self.head = nn.Linear(2688, out_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # create channel dimension

        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(batch_size, -1)
        x = self.head(x)
        return F.softmax(x, dim=1)  # softmax over classes


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    model = CNN()
    dummy_batch = torch.randn((16, 99,161))
    out = model(dummy_batch)

    print('out shape: ', out.shape)
    plt.matshow(out.detach().numpy())
    plt.colorbar()
    plt.show()

    plt.plot(out[0].detach().numpy())
    plt.show()
