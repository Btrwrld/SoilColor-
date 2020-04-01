import torch
import torch.nn as nn
import torch.nn.init as init
import time
import copy

class Image_Model(nn.Module):
    def __init__(self):
        super(Image_Model, self).__init__()

        self.conv1= nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        self.relu1= nn.ELU()
        self.norm1= nn.BatchNorm2d(64)
        self.pool1= nn.MaxPool2d(kernel_size=2, stride=2)
       
        
        self.conv2= nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.relu2= nn.ELU()
        self.norm2= nn.BatchNorm2d(32)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=2048, out_features=500) 
        self.sigm1 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=500, out_features=3)
        self.sigm2 = nn.Sigmoid() 


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)
                
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.pool2(x)
        
        x = x.view(-1,2048) 

        x = self.fc1(x)
        x = self.sigm1(x)
        x = self.fc2(x)
        x = self.sigm2(x)
        
        return x