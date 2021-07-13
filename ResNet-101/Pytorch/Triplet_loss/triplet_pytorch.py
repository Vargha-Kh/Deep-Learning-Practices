import torch
import torch.nn as nn
from model import get_model
from callback import EarlyStopping, Model_checkpoint, CSV_log
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from data_generator import DataGenerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(data, model, criterion, optimizer, device):
    loss_ = 0
    total_loss = 0
    num_image = 0
    model.train()
    for X, y in data:
        optimizer.zero_grad()
        num_image += X[0].size(0)
        x1 = X[0].to(device).float()
        x2 = X[1].to(device).float()
        x3 = X[2].to(device).float()
        x1 = model(x1)
        x2 = model(x2)
        x3 = model(x3)
        loss = criterion(x1, x2, x3)
        total_loss += loss
        loss.backward()
        optimizer.step()

    total_loss_train = total_loss / num_image
    return model, total_loss_train.item()


def valid(data, model, criterion, device):
    loss_ = 0
    total_loss = 0
    num_image = 0
    model.eval()
    for X, y in data:
        num_image += X[0].size(0)
        x1 = X[0].to(device).float()
        x2 = X[1].to(device).float()
        x3 = X[2].to(device).float()
        x1 = model(x1)
        x2 = model(x2)
        x3 = model(x3)
        loss = criterion(x1, x2, x3)
        total_loss += loss
    total_loss_valid = total_loss / num_image

    return model, total_loss_valid.item()


def training(model, ds_train, ds_valid, criterion, optimizer, scheduler, device, epochs):
    train_losses = []
    valid_losses = []

    early_stopping = EarlyStopping()
    for epoch in range(epochs):
        print("epoch:", epoch)
        model, total_loss_train = train(ds_train, model, criterion, optimizer, device)
        writer.add_scalar("train_loss", total_loss_train, epoch)
        train_losses.append(total_loss_train)
        with torch.no_grad():
            model, total_loss_valid = valid(ds_valid, model, criterion, device)
            valid_losses.append(total_loss_valid)
            writer.add_scalar("validation_loss", total_loss_valid, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(total_loss_valid)
        scores = {'epoch': epoch, 'loss': total_loss_train,
                  'val_loss': total_loss_valid, 'LR': optimizer.param_groups[0]['lr']}

        CSV_log(path=csv_log_dir, filename='log_file', score=scores)
        scheduler.step(total_loss_valid)
        metrics = {'train_loss': train_losses,  'val_loss': valid_losses}

        Model_checkpoint(path=model_checkpoint_dir, metrics=metrics, model=model,
                         monitor='val_loss', verbose=True,
                         file_name="best.pth")
        if early_stopping.Early_Stopping(monitor='val_loss', metrics=metrics, patience=3, verbose=True):
            break
        print("Epoch:", epoch + 1, "- Train Loss:", total_loss_train, "- Validation Loss:", total_loss_valid)
    return model, optimizer, train_losses, valid_losses


# Data Loading
train_get = DataGenerator(img_dir='/home/vargha/Desktop/data', base_dir='..', batch_size=1)
train_data = torch.utils.data.DataLoader(train_get, batch_size=1, shuffle=True)
valid_get = DataGenerator(img_dir='/home/vargha/Desktop/data', base_dir='..', batch_size=1, valid=True)
valid_data = torch.utils.data.DataLoader(valid_get, batch_size=1, shuffle=True)
v = 10

model_checkpoint_dir = "/home/vargha/Desktop/Vargha/Project/Triplet_loss/model"
csv_log_dir = "/home/vargha/Desktop/Vargha/Project/Triplet_loss/csv_log"

triplet_loss_list = []
pre_loss_list = []
loss_list = []

logdir = "logs/scalars/"
writer = SummaryWriter(logdir)

model = get_model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion_ce = nn.CrossEntropyLoss()
criterion = torch.nn.TripletMarginLoss(margin=0.4)
reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model, optimizer, train_loss, valid_loss = training(model, train_data, valid_data, criterion, optimizer,
                                                    reduce_on_plateau, device, 50)

