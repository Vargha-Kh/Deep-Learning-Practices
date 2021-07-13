import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


class resnet_model(nn.Module):
    def __init__(self, cut_at_pooling=False, num_features=0, norm=False, dropout=0, num_classes=0):
        super(resnet_model, self).__init__()
        self.cut_at_pooling = cut_at_pooling
        resnet = models.resnet101(pretrained=False)
        resnet.load_state_dict(torch.load('/home/vargha/Desktop/Vargha/Pytorch/model/best.pth'), strict=False)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                nn.init.kaiming_normal_(self.feat.weight, mode='fan_out')
                nn.init.constant_(self.feat.bias, 0)
            else:
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.feat_bn.weight, 1)
        nn.init.constant_(self.feat_bn.bias, 0)

    def forward(self, x, feature_withbn=False):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x, prob
        return x, prob
