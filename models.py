import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import datetime
import os
from os.path import join

class CheckpointSaver(object):
    def __init__(self, root='checkpoints', run=None, model_name='CNN_speech'):
        if not os.path.exists(root):
            os.makedirs(root) 

        if run is None:
            now = datetime.datetime.now()
            current_time = now.strftime('%b%d_%H-%M-%S')
            current_time = os.path.join(root, current_time)
            save_dir = os.path.join(root, current_time)
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

class ResnetConvLayers(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetConvLayers, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=pretrained) # pretrained ImageNet ResNet-101
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.frozen = False

    def forward(self, x):
        return self.resnet(x)

    def freeze(self):
        if self.frozen:
            return None
        self.frozen = True
        # Freeze weights
        resnet_params = []
        for param in self.resnet.parameters():
            param.requires_grad = False
            resnet_params.append(param)
        return True


class ResnetCNN(nn.Module):
    def __init__(self, input_dims=(99, 161), out_classes=30, hidden_channels=128):
        super(ResnetCNN, self).__init__()

        self.resnet_base = ResnetConvLayers()  # pretrained weights
        self.head = nn.Linear(49152, out_classes)

    def forward(self, x):
        batch_size, w, h = x.shape[0], x.shape[2], x.shape[3]
        x = x.expand(batch_size, 3, w, h)  # create channel dimension

        x = self.resnet_base(x)
        x = torch.tanh(x)
        x = x.view(batch_size, -1)
        x = self.head(x)
        return torch.softmax(x, dim=1)  # softmax over classes


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    model = CNN()
    model.resnet_base.freeze()  # model.resnet_base.frozen = True

    dummy_batch = torch.randn((16, 1, 99,161))
    out = model(dummy_batch)
    print('out shape: ', out.shape)

    plt.matshow(out.detach().numpy())
    plt.colorbar()
    plt.show()

    plt.plot(out[0].detach().numpy())
    plt.show()
