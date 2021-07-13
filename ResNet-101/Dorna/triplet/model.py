import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True


class Triple(nn.Module):
    def __init__(self, pretrained_model, vecdim=256):
        super(Triple, self).__init__()
        self.num_features = list((list(pretrained_model.children()))[-2][-1].children())[-2].num_features
        self.model_pt = pretrained_model
        self.LN = nn.LayerNorm(self.num_features)
        self.dense = nn.Linear(in_features=self.num_features, out_features=vecdim,bias=False)

    def forward(self, x1):
        x1 = self.model_pt(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.LN(x1)
        x1 = self.dense(x1)
        x1 = F.normalize(x1, dim=-1)
        return x1


# class Tirp(nn.Module):
#     def __init__(self, pretrained_model):
#         super(Tirp, self).__init__()
#         self.m1 = Triple(pretrained_model)
#         self.m2 = Triple(pretrained_model)
#         self.m3 = Triple(pretrained_model)
#
#     def forward(self, x1, x2, x3):
#         x1 = self.m1(x1)
#         x2 = self.m1(x2)
#         x3 = self.m1(x3)
#         x = torch.cat([x1, x2, x3], axis=-1)
#         return x


def get_model():
    model_pt = models.resnet101(pretrained=True)
    newmodel = nn.Sequential(*(list(model_pt.children())[:-1]))
    set_parameter_requires_grad(newmodel)
    triple_model = Triple(newmodel)
    return triple_model


# a, b = get_model()
# a = a.to(device)
# b = b.to(device)
# summary(a, (3, 224, 224))
# summary(b, (3, 224, 224))
# v = 10
