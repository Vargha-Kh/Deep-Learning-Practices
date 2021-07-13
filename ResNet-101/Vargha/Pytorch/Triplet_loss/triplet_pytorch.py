import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import resnet_model
from torch.optim import lr_scheduler
import loss

transform_list = [
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_compose = transforms.Compose(transform_list)

try_dataset1 = datasets.ImageFolder('/home/vargha/Desktop/data/train', transform_compose)
try_dataloader1 = torch.utils.data.DataLoader(try_dataset1, batch_size=16, shuffle=True)

try_data1_len = len(try_dataset1)
try_data1_class_name = try_dataset1.classes

net = resnet_model(num_classes=len(try_data1_class_name))
net.cuda()

params = []
for key, value in net.named_parameters():
    if not value.requires_grad:
        continue
    params += [{'params': [value], 'lr': 0.0001, 'weight_decay': 5e-4}]

optimizer = torch.optim.SGD(params)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

criterion_ce = nn.CrossEntropyLoss()
criterion_triple = loss.TripletLoss()

triplet_loss_list = []
pre_loss_list = []
loss_list = []
acc_list = []

for epoch in range(30):

    print("epoch: {} / 30".format(epoch + 1))
    for data in try_dataloader1:
        input, labels = data
        input = input.cuda()
        labels = labels.cuda()

        features, pres = net(input)

        tri_loss, _ = criterion_triple(features, labels)
        ce_loss = criterion_ce(features, labels)

        loss = tri_loss + ce_loss
        triplet_loss_list.append(tri_loss.item())
        pre_loss_list.append(ce_loss.item())
        loss_list.append(loss.item())

        _, pid = torch.max(pres.data, dim=1)
        acc = torch.sum(pid == labels.data) / pid.size(0)
        acc_list.append(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    exp_lr_scheduler.step()
    all_acc = sum(acc_list) / (len(acc_list))
    all_triple_loss = sum(triplet_loss_list) / (len(triplet_loss_list))
    all_pre_loss = sum(pre_loss_list) / (len(pre_loss_list))
    all_loss = sum(loss_list) / (len(loss_list))

    print('accuracy: {:.4f}'.format(all_acc))
    print('triplet loss: {:.4f}:'.format(all_triple_loss))
    print('predict loss: {:.4f}'.format(all_pre_loss))
    print('loss : {:.4f}'.format(all_loss))