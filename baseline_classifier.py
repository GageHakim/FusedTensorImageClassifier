from helpers import maths
from network import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Ensure these paths match your directory structure
sys.path.append('./hific')
sys.path.append('./hific/src')


# --- 1. The Custom Lightweight Backbone (Copied from FusedClassifier) ---
class ResidualBlock(nn.Module):
    """
    Standard ResNet Block with optional downsampling.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LatentResNet(nn.Module):
    """
    Optimized for Input: (Batch, 220, 16, 16)
    """

    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super(LatentResNet, self).__init__()

        # 1. Projection Layer: 220 channels -> 128 channels
        self.conv_in = nn.Conv2d(
            in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(128)

        # 2. Residual Stages
        self.layer1 = ResidualBlock(128, 128, stride=1)
        self.layer2 = ResidualBlock(128, 128, stride=1)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 256, stride=1)
        self.layer5 = ResidualBlock(256, 512, stride=2)
        self.layer6 = ResidualBlock(512, 512, stride=1)

        # 3. Classification Head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# --- 2. The Main Baseline Classifier ---
class BaselineClassifier(nn.Module):
    """
    A baseline classifier that uses only the main latent tensor 'y' from HiFiC's encoder,
    without the hyperprior-based attention/fusion mechanism.
    """

    def __init__(self, hific_model, num_classes=6):
        super(BaselineClassifier, self).__init__()

        # --- Feature Extraction Component ---
        self.encoder = hific_model.Encoder

        # Freeze HiFiC Encoder (Strictly required)
        self._freeze_module(self.encoder)

        # --- Classification Head ---
        latent_channels = hific_model.args.latent_channels
        print(
            f"Initializing Custom LatentResNet for {latent_channels} channels...")
        self.classifier = LatentResNet(
            in_channels=latent_channels, num_classes=num_classes)

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def forward(self, x):
        """
        Full forward pass for training and evaluation.
        """
        # 1. Extract latents (without gradients)
        with torch.no_grad():
            y = self.encoder(x)

        # 2. Classify using the latent tensor
        # Gradients will flow from here back to the classification head
        logits = self.classifier(y.detach())
        return logits
