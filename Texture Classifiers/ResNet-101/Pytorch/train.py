from model_pytorch import get_model
from callback import EarlyStopping, Model_checkpoint, CSV_log
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from augmentation import Aug

num_classes = 7
augmentation = True
batch_size = 32
num_epochs = 300
lr = 1e-3
model_checkpoint_dir = "/home/vargha/Desktop/Vargha/Pytorch/model"
csv_log_dir = "/home/vargha/Desktop/Vargha/Pytorch/csv_log"
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
    early_stopping = EarlyStopping()
    for epoch in range(epochs):
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

        scores = {'epoch': epoch, 'acc': total_acc_train, 'loss': total_loss_train, 'val_acc': total_acc_valid,
                  'val_loss': total_loss_valid, 'LR': optimizer.param_groups[0]['lr']}

        CSV_log(path=csv_log_dir, filename='log_file', score=scores)
        scheduler.step(total_loss_valid)
        metrics = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': valid_losses, 'val_acc': valid_accs}

        Model_checkpoint(path=model_checkpoint_dir, metrics=metrics, model=model,
                         monitor='val_acc', verbose=True,
                         file_name="best.pth")
        if early_stopping.Early_Stopping(monitor='val_acc', metrics=metrics, patience=3, verbose=True):
            break
        print("Epoch:", epoch + 1, "- Train Loss:", total_loss_train, "- Train Accuracy:", total_acc_train,
              "- Validation Loss:", total_loss_valid, "- Validation Accuracy:", total_acc_valid)
    return model, optimizer, train_losses, valid_losses


data_dir = '/home/vargha/Desktop/data'

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
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

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

model = get_model(num_classes)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
logdir = "logs/scalars/"
writer = SummaryWriter(logdir)
model, optimizer, train_loss, valid_loss = training(model, trainloader, testloader, criterion, optimizer,
                                                    reduce_on_plateau, device, 50)
