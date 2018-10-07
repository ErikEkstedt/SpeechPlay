from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import WordClassificationDataset, collate_fn
from torch.utils.data import DataLoader
from models import ResnetCNN, CheckpointSaver

from tensorboardX import SummaryWriter
writer = SummaryWriter()


# HyperParameters
pretrained = False
batch_size = 64
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
model = ResnetCNN(pretrained=pretrained).to(device)
model.resnet_base.freeze()
optimizer = optim.Adam(model.head.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

model.train()

for epoch in range(0, n_epochs):
    epoch_loss = 0
    for d in tqdm(dloader, desc="Epoch {}/{}".format(epoch, n_epochs)):
        # samples = d['samples']
        log_specs = d['log_specs'].to(device)
        log_specs = log_specs.unsqueeze(1)  # add channel dimension
        labels = d['labels'].to(device)

        optimizer.zero_grad()
        out = model(log_specs)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print('Epoch Loss: ', epoch_loss)
    checkpoints.save(model, optimizer, epoch, epoch_loss)
    writer.add_scalar('Loss', epoch_loss, epoch)
