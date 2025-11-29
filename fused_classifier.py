import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Ensure these paths match your directory structure
sys.path.append('./hific')
sys.path.append('./hific/src')
from network import encoder
from network import hyper
from helpers import maths


# --- 1. The Custom Lightweight Backbone ---
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
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
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
        # We reduce channels slightly to control parameter count,
        # but keep spatial dim 16x16.
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(128)

        # 2. Residual Stages
        # Stage 1: 16x16 resolution
        self.layer1 = ResidualBlock(128, 128, stride=1)
        self.layer2 = ResidualBlock(128, 128, stride=1)

        # Stage 2: Downsample to 8x8
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 256, stride=1)

        # Stage 3: Downsample to 4x4 (Final semantic features)
        self.layer5 = ResidualBlock(256, 512, stride=2)
        self.layer6 = ResidualBlock(512, 512, stride=1)

        # 3. Classification Head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights (He initialization for ReLU networks)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: [B, 220, 16, 16]
        out = F.relu(self.bn_in(self.conv_in(x)))  # -> [B, 128, 16, 16]

        out = self.layer1(out)
        out = self.layer2(out)  # -> [B, 128, 16, 16]

        out = self.layer3(out)
        out = self.layer4(out)  # -> [B, 256, 8, 8]

        out = self.layer5(out)
        out = self.layer6(out)  # -> [B, 512, 4, 4]

        out = self.avg_pool(out)  # -> [B, 512, 1, 1]
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dropout(out)
        out = self.fc(out)
        return out


# --- 2. The Gated Fusion Module (Unchanged) ---
class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        gate = torch.sigmoid(self.gate_conv(x))
        return gate * x1 + (1 - gate) * x2


# --- 3. The Main Fused Classifier ---
class FusedClassifier(nn.Module):
    def __init__(self, hific_model, num_classes=6):
        super(FusedClassifier, self).__init__()

        # --- Feature Extraction Components ---
        self.encoder = hific_model.Encoder
        self.analysis_net = hific_model.Hyperprior.analysis_net
        self.synthesis_mu = hific_model.Hyperprior.synthesis_mu
        self.synthesis_std = hific_model.Hyperprior.synthesis_std

        # Freeze HiFiC (Strictly required)
        self._freeze_module(self.encoder)
        self._freeze_module(self.analysis_net)
        self._freeze_module(self.synthesis_mu)
        self._freeze_module(self.synthesis_std)

        # HiFiC typically uses 220 latent channels
        latent_channels = hific_model.args.latent_channels
        self.fusion = GatedFusion(latent_channels)

        # --- Replaced EfficientNet with LatentResNet ---
        print(f"Initializing Custom LatentResNet for {latent_channels} channels...")
        self.classifier = LatentResNet(in_channels=latent_channels, num_classes=num_classes)

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def forward(self, x):
        """
        For compatibility with training loop.
        """
        features = self.extract_features(x)
        logits = self.classify_features(features)
        return logits

    def extract_features(self, x):
        """
        Runs the HiFiC encoder part of the model.
        """
        with torch.no_grad():
            y = self.encoder(x)
            z = self.analysis_net(y)
            z_quantized = torch.round(z)
            latent_scales = self.synthesis_std(z_quantized)
            latent_scales = maths.LowerBoundToward.apply(latent_scales, 0.11)
        return y, latent_scales

    def classify_features(self, features):
        """
        Runs the fusion and classification head.
        """
        y, latent_scales = features
        # Detach inputs to stop backprop into HiFiC
        fused = self.fusion(y.detach(), latent_scales.detach())
        logits = self.classifier(fused)
        return logits