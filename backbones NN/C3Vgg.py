import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])


        self.de_pred = nn.Sequential(nn.Conv2d(512, 128, 1), nn.ReLU(),
                                     nn.Conv2d(128, 1, 1), nn.ReLU())



    def forward(self, x):
        x = self.features4(x)       
        x = self.de_pred(x)

        x = F.upsample(x,scale_factor=2)

        return x
