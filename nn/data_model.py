import torch 
import torch.nn as nn
import torch.nn.functional as F 


class Data_Model(nn.Module):


    def __init__(self):
        super(Data_Model, self).__init__()

        # Input is a feature vector of 1x12 and we want a 1x60 output vector
        self.hidden = nn.Linear(in_features=12, out_features=60)
        self.sig1 = nn.Sigmoid()
        # And we want a 1x3 output so we can make the regression 
        self.output = nn.Linear(in_features=60, out_features=3)
        self.sig2 = nn.Sigmoid()


    def forward(self, x):

        # Since we are trying to imitate the matlab Feedforward Neural Network
        # we'll use a sigmoid activation function here
        x = self.hidden(x)
        x = self.sig1(x)
        # Then a sigmoid in the output
        x = self.output(x)
        x = self.sig2(x)

        return x   

