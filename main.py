from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import WordClassificationDataset, collate_fn
from torch.utils.data import DataLoader
from models import ResnetCNN, CheckpointSaver, ConvBatchNorm

from tensorboardX import SummaryWriter
writer = SummaryWriter()


def train(model, loss_fn, optimizer, dloader, val_loader, n_epochs):
    for epoch in range(0, n_epochs):
        train_loss = train_epoch(model, loss_fn, optimizer, dloader, epoch, n_epochs)
        val_loss, val_accuracy = evaluation(epoch, model, loss_fn, val_loader)
        print('Loss: ', train_loss)
        print('Val: ', val_loss)
        print('Acc: ', val_accuracy)
        checkpoints.save(model, optimizer, epoch, val_loss)
        writer.add_scalars('Loss', {'Training': train_loss,
                                   'Validation': val_loss}, epoch)
        writer.add_scalar('Accuracy', val_accuracy, epoch)


def train_epoch(model, loss_fn, optimizer, dloader, epoch, n_epochs):
    model.train()
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
    return epoch_loss


def evaluation(epoch, model, loss_fn, dloader):
    model.eval()
    total, correct, eval_loss = 0, 0, 0
    with torch.no_grad():
        for d in tqdm(dloader, desc="Val"):
            # samples = d['samples']
            log_specs = d['log_specs'].to(device)
            labels = d['labels'].to(device)

            out = model(log_specs)
            best_idx = out.argmax(dim=1).long()

            loss = loss_fn(out, labels)
            eval_loss += loss.item()

            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return eval_loss, correct/total


# TODO
def eval_classes():
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


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

n_classes = 30
in_channels = 1
filters = [32, 32, 64, 64, 128, 128]
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
train(model, loss_fn, optimizer, dloader, dloader, n_epochs)
