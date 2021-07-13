import torch
import torch.nn as nn
from model import get_model
from torchvision import datasets, transforms
# from model import resnet_model
from torch.optim import lr_scheduler
import loss
from data_generatot import DataGenerator
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# transform_compose = transforms.Compose([
#     transforms.Resize((224, 224), interpolation=3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# try_dataset1 = datasets.ImageFolder('/home/vargha/Desktop/data/train', transform_compose)
# = torch.utils.data.DataLoader(try_dataset1, batch_size=16, shuffle=True)
# try_dataloader1 = DataGenerator(img_dir='/home/dorna/symo/Dorna/triplet_loss/data', base_dir='..', batch_size=32)

# try_data1_len = len(try_dataset1)
# try_data1_class_name = try_dataset1.classes
train_get = DataGenerator(img_dir='/home/dorna/symo/Dorna/triplet_loss/data', base_dir='..', batch_size=32)
train_data= torch.utils.data.DataLoader(train_get, batch_size=32, shuffle=True)
valid_get = DataGenerator(img_dir='/home/dorna/symo/Dorna/triplet_loss/data', base_dir='..', batch_size=32, valid=True)
valid_data= torch.utils.data.DataLoader(valid_get, batch_size=32, shuffle=True)
v = 10


def train(data, model, criterion, optimizer, device):
    loss_ = 0
    total_loss = 0
    num_image = 0
    model.train()
    for X, y in data:
        optimizer.zero_grad()
        num_image += X[0].size(0)
        # anchor_img, pos_img, neg_img = img_triplet
        # anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        # anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        # E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        # dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        # dist_E1_E3 = F.pairwise_distance(E1, E3, 2)
        #
        # target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        # target = target.to(device)
        # target = Variable(target)
        x1 = X[0].to(device)
        x2 = X[1].to(device)
        x3 = X[2].to(device)
        x1 = model(x1)
        x2 = model(x2)
        x3 = model(x3)
        loss = criterion(x1, x2, x3)
        total_loss += loss
        loss.backward()
        optimizer.step()

    total_loss_train = loss_ / num_image

    return model, total_loss_train


def valid(data, model, criterion, device):
    loss_ = 0
    total_loss = 0
    num_image = 0
    model.train()
    for X, y in data:
        num_image += X.size(0)
        # anchor_img, pos_img, neg_img = img_triplet
        # anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        # anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        # E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        # dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        # dist_E1_E3 = F.pairwise_distance(E1, E3, 2)
        #
        # target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        # target = target.to(device)
        # target = Variable(target)
        x1 = X[0].to(device)
        x2 = X[1].to(device)
        x3 = X[2].to(device)
        x1 = model(x1)
        x2 = model(x2)
        x3 = model(x3)
        loss = criterion(x1, x2, x3)
        total_loss += loss
    total_loss_valid = loss_ / num_image

    return model, total_loss_valid


model = get_model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

criterion_ce = nn.CrossEntropyLoss()
# criterion_triple = loss.TripletLoss()
criterion = torch.nn.TripletMarginLoss(margin=0.4)
triplet_loss_list = []
pre_loss_list = []
loss_list = []
acc_list = []


def training(model, ds_train, ds_valid, criterion, optimizer, scheduler, device, epochs):
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        print("epoch=", epoch)
        model, total_loss_train = train(ds_train, model, criterion, optimizer, device)
        writer.add_scalar("train_loss", total_loss_train, epoch)
        train_losses.append(total_loss_train)
        with torch.no_grad():
            model, total_loss_valid = valid(ds_valid, model, criterion, device)
            valid_losses.append(total_loss_valid)
            writer.add_scalar("validation_loss", total_loss_valid, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        # CSVLogger('/csv_log')
        scheduler.step()
        # Model_checkpoint("/saved", model, valid_losses)
        # if Early_Stopping:
        #     break
        # else:
        #     continue

        print("epoch=", epoch + 1)
        print("train loss=", total_loss_train)
        print("valid loss=", total_loss_valid)

    return model, optimizer, train_losses, valid_losses


# for epoch in range(30):
#
#     print("epoch: {} / 30".format(epoch + 1))
#     for data in try_dataloader1:
#         input, labels = data
#         input = input.cuda()
#         labels = labels.cuda()
#
#         features, pres = net(input)
#
#         tri_loss, _ = criterion_triple(features, labels)
#         ce_loss = criterion_ce(features, labels)
#
#         loss = tri_loss + ce_loss
#         triplet_loss_list.append(tri_loss.item())
#         pre_loss_list.append(ce_loss.item())
#         loss_list.append(loss.item())
#
#         _, pid = torch.max(pres.data, dim=1)
#         acc = torch.sum(pid == labels.data) / pid.size(0)
#         acc_list.append(acc.item())
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     exp_lr_scheduler.step()
#     all_acc = sum(acc_list) / (len(acc_list))
#     all_triple_loss = sum(triplet_loss_list) / (len(triplet_loss_list))
#     all_pre_loss = sum(pre_loss_list) / (len(pre_loss_list))
#     all_loss = sum(loss_list) / (len(loss_list))
#
#     print('accuracy: {:.4f}'.format(all_acc))
#     print('triplet loss: {:.4f}:'.format(all_triple_loss))
#     print('predict loss: {:.4f}'.format(all_pre_loss))
#     print('loss : {:.4f}'.format(all_loss))
reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
model, optimizer, train_loss, valid_loss = training(model, train_data, valid_data, criterion, optimizer,
                                                    reduce_on_plateau, device, 50)
v1 = 10
