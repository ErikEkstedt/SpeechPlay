from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import WordClassificationDataset, collate_fn
from torch.utils.data import DataLoader
from models import CNN, CheckpointSaver

from tensorboardX import SummaryWriter
writer = SummaryWriter()


# HyperParameters
batch_size = 256
n_epochs = 50
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
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

for epoch in range(0, n_epochs):
    epoch_loss = 0
    for d in tqdm(dloader, desc="Epoch {}/{}".format(epoch, n_epochs)):
        # samples = d['samples']
        log_specs = d['log_specs'].to(device)
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
