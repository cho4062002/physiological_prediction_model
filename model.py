import torch.nn as nn
import torch
import numpy

class HR_BR_net(nn.Module):
    def __init__(self):
        super(HR_BR_net, self).__init__()
        self.a_conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.a_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.a_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.a_conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.a_avgpool = nn.AvgPool2d(2)

        self.a_conv2_ = nn.Conv2d(32, 32 ,1)

        self.a_conv4_ = nn.Conv2d(64, 64 ,1)

        self.a_conv_norm = nn.L1Loss
        
        self.f_conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.f_conv2 = nn.Conv2d(3, 32, 3, padding=1)

        self.f_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.f_conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.f_avgpool = nn.AvgPool2d(2)

        ### self.fc1 = nn.Linear() 
        ### self.fc2 = nn.Linear() 

def forward(self, appear, feature):
    appear = self.a_conv1(appear)

    appear = self.a_conv2(appear)

    appear = self.a_avgpool(appear)

    appear_norm_1 = self.a_conv2_(appear)

    appear_norm_1 = self.a_conv_norm(appear_norm_1)

    appear = self.a_conv3(appear)

    appear = self.a_conv4(appear)

    appear_norm_2 = self.a_conv4_(appear)

    appear_norm_2 = self.a_conv_norm(appear_norm_2)

    feature = self.f_conv1(feature)

    feature = self.f_conv2(feature)

    feature = torch.mul(feature, appear_norm_1)

    feature = self.f_avgpool(feature)

    feature = self.f_conv3(feature)

    feature = self.f_conv4(feature)

    feature = torch.mul(feature, appear_norm_2)

    feature = self.f_avgpool(feature)

    feature = feature.view(feature.size(0), -1)

    output = self.fc(feature)

    return output


    

      







