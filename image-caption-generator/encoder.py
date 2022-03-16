import torch
import torch.nn as nn

class CNN_Encoder(torch.nn.Module):
    """
    Basic logistic regression on 3x244x244 images.
    """

    def __init__(self, units, embedding_dim, dropout_pct=0, 
                 model_name='resnet18', freeze_CNN=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3)
        tempmodel = torch.hub.load('pytorch/vision:v0.9.0', model_name, 
                                   pretrained=True)

        if freeze_CNN: # freezing the transfer learning model params
          for param in tempmodel.parameters():
            param.requires_grad = False
        
        self.resnet = torch.nn.Sequential(*(list(tempmodel.children())[:-1]))
        self.fc1 = nn.Linear(units, units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_pct)
        self.bnfc1 = nn.BatchNorm1d(units)
        self.fc2 = nn.Linear(units, embedding_dim)


    def forward(self, x):
        unsqueeze_bool = False
        if(x.shape[0]) == 1:
          unsqueeze_bool = True
        x = self.relu(self.conv(x))
        x = self.resnet(x)
        x = torch.squeeze(x)  # flatten
        if unsqueeze_bool:
          x = torch.unsqueeze(x, 0)
        x = self.fc1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.bnfc1(self.dropout(x))
        x = self.fc2(x)
        return x