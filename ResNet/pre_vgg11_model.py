# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.models as models

# Define the VGG11Transfer model
class VGG11Transfer(nn.Module):
    def __init__(self, num_classes):
        super(VGG11Transfer, self).__init__()
        self.features = models.vgg11(pretrained=True).features
        self.features[0] = torch.nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),

            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        print(self.features)
        weight = self.features.state_dict()
        first_layer_weight = weight['0.weight']
        second_layer_weight = weight['3.weight']
        # print("第1层权重：", first_layer_weight)
        # print("第1层权重形状：", first_layer_weight.shape)
        # print("第2层权重：", second_layer_weight)
        # print("第2层权重形状：", second_layer_weight.shape)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
