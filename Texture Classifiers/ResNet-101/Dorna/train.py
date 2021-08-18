# from callback import Early_Stopping, Model_checkpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from augmentation import Aug
from model_pytorch import get_model

# from pytorchtools import EarlyStopping
# from callbacks.csv_logger import CSVLogger
# from torchtrainer.callbacks import CSVLogger
augmentation = True
num_classes = 7
batch_size = 32
num_epochs = 300
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(ds_train, model, criterion, optimizer, device):
    model.train()

    loss_ = 0
    train_acc = 0
    num_image = 0
    for x, y_true in ds_train:
        optimizer.zero_grad()
        X = x.to(device)
        Y = y_true.to(device)
        logit, probe = model(X)
        loss = criterion(logit, Y)
        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        train_acc += torch.sum(num == Y)
        num_image += x.size(0)
        loss.backward()
        optimizer.step()
    total_loss_train = loss_ / num_image
    total_acc_train = train_acc / num_image

    return model, total_loss_train, total_acc_train.item()


def valid(ds_valid, model, criterion, device):
    model.eval()
    loss_ = 0
    valid_acc = 0
    num_image = 0
    for x, y_true in ds_valid:
        X = x.to(device)
        Y = y_true.to(device)
        logit, probe = model(X)
        loss = criterion(logit, Y)

        loss_ += loss.item() * x.size(0)
        Max, num = torch.max(logit, 1)
        valid_acc += torch.sum(num == Y)
        num_image += x.size(0)
    total_loss_valid = loss_ / num_image
    total_acc_valid = valid_acc / num_image
    return model, total_loss_valid, total_acc_valid.item()


def training(model, ds_train, ds_valid, criterion, optimizer, scheduler, device, epochs):
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for epoch in range(epochs):
        print("epoch=", epoch)
        model, total_loss_train, total_acc_train = train(ds_train, model, criterion, optimizer, device)
        writer.add_scalar("train_loss", total_loss_train, epoch)
        writer.add_scalar("train_accuracy", total_acc_train, epoch)
        train_losses.append(total_loss_train)
        train_accs.append(total_acc_train)
        with torch.no_grad():
            model, total_loss_valid, total_acc_valid = valid(ds_valid, model, criterion, device)
            valid_losses.append(total_loss_valid)
            valid_accs.append(total_acc_valid)
            writer.add_scalar("validation_loss", total_loss_valid, epoch)
            writer.add_scalar("validation_accuracy", total_acc_valid, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        CSVLogger('/csv_log')
        scheduler.step()
        Model_checkpoint("/saved", model, valid_losses)
        if Early_Stopping:
            break
        else:
            continue

        print("epoch=", epoch + 1)
        print("train loss=", total_loss_train)
        print("train acc=", total_acc_train)
        print("valid loss=", total_loss_valid)
        print("valid acc=", total_acc_valid)
    return model, optimizer, train_losses, valid_losses


data_dir = '/home/dorna/symo/Dorna/data'

train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
if augmentation:
    augmented_data = Aug(data_dir + "/train")
    augmented_data.append(train_data)
    train_data = torch.utils.data.ConcatDataset(augmented_data)
# train_transforms(train_datas)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

model = get_model(num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
# csv_logger = CSVLogger(file=csv_dir)
logdir = "logs/scalars/"
writer = SummaryWriter(logdir)
model, optimizer, train_loss, valid_loss = training(model, trainloader, testloader, criterion, optimizer,
                                                    reduce_on_plateau, device, 50)
