import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
