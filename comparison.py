import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics import classification_report
import numpy as np

# Define the model architecture (use your real one)
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x)

# Load function
def load_model(state_path):
    model = CustomModel()
    state_dict = torch.load(state_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# File paths
model1_path = 'model_epoch_20.pth'
model2_path = 'Xception-598X598.pth'

# Load both models
model1 = load_model(model1_path)
model2 = load_model(model2_path)
