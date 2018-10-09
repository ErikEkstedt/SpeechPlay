from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import WordClassificationDataset, collate_fn
from torch.utils.data import DataLoader
from models import ResnetCNN, CheckpointSaver, ConvBatchNorm

from tensorboardX import SummaryWriter
writer = SummaryWriter()

resnet = False
conv_batch = True
batch_size = 256
n_epochs = 500
num_workers = 4
pin_memory = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Load Dataset')
dset = WordClassificationDataset()
dloader = DataLoader(dset,
                     collate_fn=collate_fn,
                     batch_size=batch_size,
                     num_workers=num_workers,
                     pin_memory=pin_memory,
                     shuffle=True)

checkpoints = CheckpointSaver()
print('Saving Checkpoints to ', checkpoints.save_dir)


print('Using ', device)
# ResNet
if resnet:
    # HyperParameters, 
    pretrained = False
    model = ResnetCNN(pretrained=pretrained).to(device)
    model.resnet_base.freeze()
elif conv_batch:
    # HyperParameters,  GST reference encoder
    n_classes = 30
    in_channels = 1
    filters=[32, 32, 64, 64, 128, 128]
    kernels = [3,3,3,3,3,3]
    strides = [2,2,2,2,1,1]
    padding = [0,0,0,0,0,0]
    n_mels = 161
    model = ConvBatchNorm(n_classes,
                          in_channels,
                          filters,
                          kernels,
                          strides,
                          padding,
                          n_mels).to(device)
    params, trainable_params = model.total_parameters()
    print('total parameters: ', params)
    print('total trainable parameters: ', trainable_params)


optimizer = optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()

for epoch in range(0, n_epochs):
    epoch_loss = 0
    for d in tqdm(dloader, desc="Epoch {}/{}".format(epoch, n_epochs)):
        # samples = d['samples']
        log_specs = d['log_specs'].to(device)
        log_specs = log_specs.unsqueeze(1)  # add channel dimension
        labels = d['labels'].to(device)
        # print('spec: ', log_specs.shape)
        # print('labels: ', labels.shape)
        # input()

        optimizer.zero_grad()
        out = model(log_specs)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print('Epoch Loss: ', epoch_loss)
    checkpoints.save(model, optimizer, epoch, epoch_loss)
    writer.add_scalar('Loss', epoch_loss, epoch)
