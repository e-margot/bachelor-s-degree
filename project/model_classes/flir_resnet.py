from project.model_classes.image_classificaton_base import ImageClassificationBase
from project.model_classes.resnet import ResNet, BasicBlock
import torch.nn as nn


def resnet34():
    layers = [2, 2, 2, 2]
    model_instance = ResNet(BasicBlock, layers)
    return model_instance


class FLIR80Resnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = resnet34()
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 80)

    def forward(self, xb):
        return self.network(xb)
