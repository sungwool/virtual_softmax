import torch
import torchvision
import torch.nn as nn

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    return model

class virtual_layer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        c_in   = 512
        self.kernel = nn.Parameter(torch.rand(c_in, num_classes), requires_grad=True)
    
    def forward(self, x, y):
        WX = torch.matmul(x, self.kernel)
        
        if self.training:
            W_yi = self.kernel[:, y]
            WX_VIRT = torch.norm(W_yi.T, dim=1) * torch.norm(x, dim=1)
            WX_VIRT = torch.clip(WX_VIRT, min=1e-10, max=15.0).unsqueeze(-1)
            WX_NEW = torch.cat([WX, WX_VIRT], dim=1)
            return WX_NEW

        else:
            return WX
        