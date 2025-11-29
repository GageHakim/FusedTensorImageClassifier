import torch
import torch.nn as nn
from compressai.zoo import mbt2018_mean

# --- 1. A simpler classifier head ---

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron to classify flattened latent tensors.
    """
    def __init__(self, in_features, num_classes, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)

# --- 2. The Refigured Baseline Classifier ---

class BaselineClassifier(nn.Module):
    """
    A baseline classifier that uses the main latent tensor 'y' from a pre-trained
    Minnen2018-Mean model as its feature source, and classifies it with a simple MLP.
    """

    def __init__(self, num_classes=7, quality=3, dropout_rate=0.5):
        super(BaselineClassifier, self).__init__()

        print(f"Loading Minnen2018-Mean (Quality {quality}) from CompressAI Zoo for baseline...")
        self.backbone = mbt2018_mean(quality=quality, pretrained=True)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Determine the size of the latent tensor
        # We need to do a dummy forward pass to get the spatial dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            dummy_latent = self.backbone.g_a(dummy_input)
            
        latent_channels = dummy_latent.shape[1]
        latent_height = dummy_latent.shape[2]
        latent_width = dummy_latent.shape[3]
        in_features = latent_channels * latent_height * latent_width
        
        print(f"Backbone Latent Tensor Shape: (B, {latent_channels}, {latent_height}, {latent_width})")
        print(f"Input features to MLP: {in_features}")

        self.classifier = SimpleMLP(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        y = self.extract_features(x)
        logits = self.classify_features(y)
        return logits

    def extract_features(self, x):
        with torch.no_grad():
            return self.backbone.g_a(x)

    def classify_features(self, y):
        return self.classifier(y.detach())