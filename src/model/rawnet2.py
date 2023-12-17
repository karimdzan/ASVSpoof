import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model import BaseModel
from src.model.sinc_layer import SincLayer


class FMS(nn.Module):
    def __init__(self, dim, add = True, mul = True):
        super(FMS, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.f = nn.Sigmoid()

    def forward(self, x):
        res = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        res = self.f(self.fc(res)).view(x.size(0), x.size(1), -1)
        x = res + res * x
        return x
    

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock, self).__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_channels), 
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        )
        
        if in_channels == out_channels:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)

        self.maxpool1d = nn.MaxPool1d(3)
        self.fms = FMS(out_channels)
        
    def forward(self, x):
        out = self.head(x) + self.downsample(x)
        out = self.maxpool1d(out)
        out = self.fms(out)
        return out
    

class RawNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()
        blocks = []
        first = True
        for i in range(n_layers):
            blocks.append(Resblock(in_channels, out_channels))
            if first == True:
                in_channels = out_channels
                first = False
        self.blocks = nn.Sequential(blocks)
        
    def forward(self, x):
        return self.blocks(x)
    

class RawNet2(BaseModel):
    def __init__(self, **config):
        super().__init__()
        self.sinc_filter = SincLayer(**config["sincnet"])
        self.rawnet_block1 = RawNetBlock(**config["rawnet_block1"])
        self.rawnet_block2 = RawNetBlock(**config["rawnet_block2"])
        self.gru_preactivations = nn.Sequential(
            nn.BatchNorm1d(config["rawnet_block2"]["out_channels"]),
            nn.LeakyReLU(0.3)
        )
        self.gru = nn.GRU(**config["gru"])
        self.fc = nn.Linear(config["gru"]["hidden_size"], config["gru"]["hidden_size"])
        self.head = nn.Linear(config["gru"]["hidden_size"], 2)
        
    def forward(self, x):
        x = self.sinc_filter(x)
        x = self.rawnet_block1(torch.abs(x))
        x = self.rawnet_block2(x)
        x = self.gru_preactivations(x)
        x, _ = self.gru(x.transpose(1, 2))
        x = self.fc(x[:, -1, :])
        norm = x.norm(p=2, dim=1, keepdim=True) / 10
        x = x / norm
        x = self.head(x)
        return x