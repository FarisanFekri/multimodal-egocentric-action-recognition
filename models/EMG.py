from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):

    # network structure
    def __init__(self,num_classes=20):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1664, 832)  # Adjusted based on input size
        self.fc2 = nn.Linear(832, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        #lsls
        features = torch.flatten(x,1)
        features = F.dropout(features, p=0.5, training=self.training)
        logits = F.relu(self.fc1(features))
        logits = self.fc2(logits)
        return logits, {"features" : features}
        
    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size) 