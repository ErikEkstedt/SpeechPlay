import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision.transforms import transforms 

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
        for param in self.resnet.parameters():
            param.requires_grad = False
        return True


class ResnetCNN(nn.Module):
    def __init__(self, input_dims=(99, 161), out_classes=30,
                 hidden_channels=128, fc_hidden=512, pretrained=True):
        super(ResnetCNN, self).__init__()

        self.resnet_base = ResnetConvLayers(pretrained=pretrained)  # pretrained weights
        self.head = nn.Sequential (
            nn.Linear(49152, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, out_classes),
        )

        # resnet preprocessing
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def resnet_normalize(self, x):
        '''resnet preprocess
        Arguments:
            x: torch.tensor (batch, 1, w, h) with norm values x in range [0, 1]
        '''
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        # batch_size = x.shape[0]
        r = x[:, 0]
        g = x[:, 1]
        b = x[:, 2]
        r = (r - mean[0]) / std[0]
        g = (g - mean[1]) / std[1]
        b = (b - mean[2]) / std[2]
        return torch.stack((r,g,b), dim=1)


    def forward(self, x):
        batch_size, w, h = x.shape[0], x.shape[2], x.shape[3]
        x = x.expand(batch_size, 3, w, h)  # create channel dimension
        x = self.resnet_normalize(x)
        x = self.resnet_base(x)
        print(x.shape)
        x = torch.tanh(x)
        x = x.view(batch_size, -1)
        x = self.head(x)
        return torch.softmax(x, dim=1)  # softmax over classes


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    model = ResnetCNN()

    model.resnet_base.freeze()  # model.resnet_base.frozen = True

    dummy_batch = torch.randn((16, 1, 99,161))
    out = model(dummy_batch)
    print('out shape: ', out.shape)

    plt.matshow(out.detach().numpy())
    plt.colorbar()
    plt.show()

    plt.plot(out[0].detach().numpy())
    plt.show()
