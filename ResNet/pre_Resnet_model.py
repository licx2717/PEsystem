# Import necessary libraries
import monai
import torch
import torch.nn as nn
import torchvision.models as models

# Load the pre-trained weights
weights = torch.load("D:\\MONAI-dev\\pretrain\\resnet_34.pth")

# Create an instance of the model
model = monai.networks.nets.resnet34(num_classes=2)

# Load the weights into the model
#resnet34_pretrained=model.load_state_dict(weights)
# Define the VGG11Transfer model
class ResnetTransfer(nn.Module):
    def __init__(self, num_classes):
        super(ResnetTransfer, self).__init__()
        self.nets = model
        self.nets.conv1=nn.Conv3d(1,16,kernel_size=7,stride=2,padding=3, bias=False)
        self.bn1=nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn. MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fc = nn.Linear(200704, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out

model = ResnetTransfer(num_classes=2)
model_dict = model.state_dict()
weights = {k: v for k, v in weights.items() if k.startswith('layer')}
model_dict.update(weights)
model.load_state_dict(model_dict)