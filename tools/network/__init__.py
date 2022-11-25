import torch
import torch.nn as nn
import math
from .vgg import VGG

def create_model():
    model = VGG('VGG19')
    return model

class virtual_layer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        c_in   = 2
        self.kernel = nn.Parameter(torch.rand(c_in, num_classes), requires_grad=True)
        # self.reset_parameters()
        self.fc = nn.Linear(c_in, num_classes)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kernel.size(1))
        self.kernel.data.uniform_(-stdv, stdv)
        
    def forward(self, x, y):
        # return self.fc(x)
    
        WX = torch.matmul(x, self.kernel)
    
        margin, _ = torch.max(WX, dim=-1)
        margin = torch.mean(margin)
        
        if self.training:
            W_yi = self.kernel[:, y]
            WX_VIRT = torch.norm(W_yi.T, p=2, dim=1) * torch.norm(x, p=2, dim=1)
            WX_VIRT = torch.clip(WX_VIRT, min=1e-4, max=50.0).unsqueeze(-1)
            
            WX_NEW = torch.cat([WX, WX_VIRT], dim=1)
            return WX_NEW

        else:
            return WX