from __future__ import print_function
from __future__ import division
import torch
import torch.utils.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = 7
for i in range(0, num_classes):
    v = 10
batch_size = 16
num_epochs = 150
# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = True


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_pt = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_pt, feature_extract)
    num_features = model_ft.fc.in_features

    class New_resnet(nn.module):
        def __int__(self, pretrained_model):
            super(New_resnet, self).__init__()
            self.pretrained = pretrained_model
            self.newlayers = nn.Sequential(
                nn.BatchNorm2d(),
                nn.Linear(in_features=num_features, out_features=512),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=512, out_features=256),
                nn.BatchNorm2d(num_features=256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=256, out_features=128),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=128, out_features=num_classes)
            )

        def forward(self, x):
            x = self.pretrained(x)
            logit = self.newlayers(x)
            prob = F.softmax(logit, dim=1)
            return logit, prob

        model = New_resnet()

    return model, input_size


model, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)
