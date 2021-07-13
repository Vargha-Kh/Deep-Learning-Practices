import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def set_parameter_requires_grad(model, fine_tune=True):
#     if fine_tune:
#         for param in model.parameters():
#             param.requires_grad = fine_tune


class Triple(nn.Module):
    def __init__(self, pretrained_model, vecdim=256):
        super(Triple, self).__init__()
        self.num_features = list((list(pretrained_model.children()))[-2][-1].children())[-2].num_features
        self.model_pt = pretrained_model
        self.LN = nn.LayerNorm(self.num_features)
        self.dense = nn.Linear(in_features=self.num_features, out_features=vecdim, bias=False)

    def forward(self, x1):
        x1 = self.model_pt(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.LN(x1)
        x1 = self.dense(x1)
        x1 = F.normalize(x1, dim=-1)
        return x1


def get_model(fine_tune=True):
    model_pt = models.resnet101(pretrained=True)
    for param in model_pt.parameters():
        param.requires_grad = False
    newmodel = nn.Sequential(*(list(model_pt.children())[:-1]))
    if fine_tune:
        for param in newmodel.parameters():
            param.requires_grad = True
    # set_parameter_requires_grad(newmodel, fine_tune=True)
    triple_model = Triple(newmodel)
    return triple_model
