import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.zoo import mbt2018_mean


# --- 1. The Lightweight Backbone (Optimized for 16x16 inputs) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super(LatentResNet, self).__init__()

        # Minnen2018-Mean (Quality 3) usually outputs 192 or 320 channels.
        # We project this down to 128 to keep the classifier light.
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        # 16x16 -> 16x16
        self.layer1 = ResidualBlock(128, 128, stride=1)
        self.layer2 = ResidualBlock(128, 128, stride=1)

        # 16x16 -> 8x8
        self.layer3 = ResidualBlock(128, 256, stride=2)

        # 8x8 -> 4x4
        self.layer4 = ResidualBlock(256, 512, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# --- 2. The Improved Baseline Classifier ---
class ImprovedBaselineClassifier(nn.Module):
    def __init__(self, num_classes=7, quality=3, dropout_rate=0.5):
        super(ImprovedBaselineClassifier, self).__init__()

        print(f"Loading Minnen2018-Mean (Quality {quality}) from CompressAI Zoo...")
        self.backbone = mbt2018_mean(quality=quality, pretrained=True)

        # FREEZE THE BACKBONE
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.latent_channels = self.backbone.M
        print(f"Backbone Latent Channels: {self.latent_channels}")

        self.classifier = LatentResNet(
            in_channels=self.latent_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classify_features(features)
        return logits

    def extract_features(self, x):
        """Extracts latent features from the image."""
        with torch.no_grad():
            y = self.backbone.g_a(x)
        return y.detach()

    def classify_features(self, y):
        """Classifies features to produce logits."""
        logits = self.classifier(y)
        return logits
