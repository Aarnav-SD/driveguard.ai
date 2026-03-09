import torch
import torch.nn as nn

class CognitiveNet(nn.Module):

    def __init__(self):

        super(CognitiveNet,self).__init__()

        self.lstm = nn.LSTM(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64,3)

    def forward(self,x):

        out,_ = self.lstm(x)

        out = out[:,-1,:]

        out = self.fc(out)

        return out