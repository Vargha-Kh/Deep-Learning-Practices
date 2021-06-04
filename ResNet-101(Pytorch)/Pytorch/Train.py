from torchvision import models
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from torch.optim import lr_scheduler
from callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard

num_classes = 7
batch_size = 32
num_epochs = 300
lr = 1e-3
# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = True


def train(ds_train, model, criterion, optimizer, device):
    model.train()

    loss_ = 0
    train_acc = 0
    c = 0
    for x, y_true in ds_train:
        optimizer.zero_grad()
        X = x.to(device)
        Y = y_true.to(device)
        logit = model(X)
        loss = criterion(logit, Y)
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        train_acc += torch.sum(num == Y)
        c += x.size(0)
        loss.backward()
        optimizer.step()
    total_loss_train = loss_ / c
    total_acc_train = train_acc / c
    return model, total_loss_train, total_acc_train.item()


def valid(ds_valid, model, criterion, device):
    model.eval()
    loss_ = 0
    valid_acc = 0
    c = 0
    for x, y_true in ds_valid:
        X = x.to(device)
        Y = y_true.to(device)
        logit = model(X)
        loss = criterion(logit, Y)
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        valid_acc += torch.sum(num == Y)
        c += x.size(0)
    total_loss_valid = loss_ / c
    total_acc_valid = valid_acc / c
    return model, total_loss_valid, total_acc_valid.item()


def training(model, ds_train, ds_valid, criterion, optimizer, device, epochs):
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        model, total_loss_train, total_acc_train = train(ds_train, model, criterion, optimizer, device)
        train_losses.append(total_loss_train)
        with torch.no_grad():
            model, total_loss_valid, total_acc_valid = valid(ds_valid, model, criterion, device)
            valid_losses.append(total_loss_valid)
        exp_lr_scheduler.step()
        print("epoch=", epoch + 1)
        print("train loss=", total_loss_train)
        print("train acc=", total_acc_train)
        print("valid loss=", total_loss_valid)
        print("valid acc=", total_acc_valid)
    return model, optimizer, train_losses, valid_losses

criterion = nn.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
