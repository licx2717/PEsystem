import torch
from ResNet.pre_Resnet_model import ResnetTransfer
from ResNet.resnet_models import resnet34

#model = ResnetTransfer(num_classes=2)
#print(model)
import monai
model=resnet34(num_classes=2)
weights = torch.load("D:\\MONAI-dev\\pretrain\\resnet_34_23dataset.pth")
# Assuming the dictionary file is named "weights"
weights = weights['state_dict']
weights = {k.replace("module.", ""): v for k, v in weights.items()}
#weights = {k.replace("device='cuda:0'", "device='cuda:1'"): v for k, v in weights.items()}
model_dict = model.state_dict()
model_dict.update(weights)
model.load_state_dict(model_dict)
print(model)
