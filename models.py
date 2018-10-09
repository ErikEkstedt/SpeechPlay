import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision.transforms import transforms 

import datetime
import os
from os.path import join

def plot_spec(s):
    if isinstance(s, torch.Tensor):
        s = s.numpy()
    plt.matshow(s.T)
    ax = plt.gca()
    plt.gca().invert_yaxis()
    plt.pause(0.1)


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
        return self.head(x)  # Scores not probs. Using CrossEntropyLoss.



# -----------------------------------------------------

class ConvBatchNorm(nn.Module):
    def __init__(self,
                 n_classes=30,
                 in_channels=1,
                 filters=[32, 32, 64, 64, 128, 128],
                 kernels=[3, 3, 3, 3, 3, 3],
                 strides=[2, 2, 2, 2, 1, 1],
                 padding=[0, 0, 0, 0, 0, 0],
                 n_mels=128):
        super(ConvBatchNorm, self).__init__()
        filters = [in_channels] + filters  # add input channel

        self.n_classes = n_classes
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.n_mels = n_mels

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=kernels[i],
                           stride=strides[i],
                           padding=(1, 1)) for i in range(len(filters)-1)]
        self.convs = nn.ModuleList(convs)

        # Spatial Batchnorm
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filters[i]) for i
                                  in range(1, len(filters))])

        self.fc1 = nn.Linear(9856, 512)
        self.out = nn.Linear(512, n_classes)

        print('conv layers: ', len(self.convs))
        print('bn layers: ', len(self.bns))
        # self.conv_out_channels = self.calculate_channels(n_mels, 3, 2, 1, K)
        # self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
        #                   hidden_size=hp.E // 2,
        #                   batch_first=True)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        channels_in = inputs.size(1)
        time_steps = inputs.size(2)
        freq_bins = inputs.size(3)

        x = inputs
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)  # [N, 128, Ty//2^K, n_mels//2^K]

        x = self.fc1(x.view(batch_size, -1))
        x = F.relu(x)
        return self.out(x) # Pytorch nn.CrossEntropy expects scores. not probs.

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

    def total_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, total_trainable_params


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from dataset import WordClassificationDataset

    # Dataset returns numpy
    dset = WordClassificationDataset(samples=True)

    # Resnet. Really bad in this setup
    model = ResnetCNN()
    model.resnet_base.freeze()  # model.resnet_base.frozen = True

    # GST reference encoder
    n_classes = 30
    in_channels = 1
    filters=[32, 32, 64, 64, 128, 128]
    kernels = [3,3,3,3,3,3]
    strides = [2,2,2,2,1,1]
    padding = [0,0,0,0,0,0]
    n_mels = 161

    # Model. Outputs scores for classes. Probs are calculated in
    # CrossEntropyLoss
    model = ConvBatchNorm(n_classes,
                          in_channels,
                          filters,
                          kernels,
                          strides,
                          padding,
                          n_mels)
    params, trainable_params = model.total_parameters()
    print('total parameters: ', params)
    print('total trainable parameters: ', trainable_params)

    loss_fn = nn.CrossEntropyLoss()

    samples, spec, label = dset.get_random()
    plot_spec(spec) 
    spec = torch.from_numpy(spec)
    label = torch.LongTensor([label])

    dummy_batch = spec.unsqueeze(0).unsqueeze(0)
    dummy_label = label.unsqueeze(0)

    out = model(dummy_batch)
    print('out shape: ', out.shape)

    loss = loss_fn(out, label)

    plt.matshow(out.detach().numpy())
    plt.colorbar()
    plt.show()

    plt.plot(out[0].detach().numpy())
    plt.show()
