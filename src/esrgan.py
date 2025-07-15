import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# =============================
# Model Definition (simplified SRResNet)
# =============================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class SRModel(nn.Module):
    def __init__(self, num_residuals=16, upscale_factor=4):  # Keep 4×
        super(SRModel, self).__init__()

        self.conv_input = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])

        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)

        # Always define both upscaling blocks for 4x model
        self.upscale_blocks = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),  # 0
            nn.PixelShuffle(2),                            # 1
            nn.ReLU(inplace=True),                         # 2
            nn.Conv2d(64, 256, kernel_size=3, padding=1),  # 3
            nn.PixelShuffle(2),                            # 4
            nn.ReLU(inplace=True)                          # 5
        )

        self.conv_output = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual_blocks(out)
        out = self.bn_mid(self.conv_mid(out)) + residual
        out = self.upscale_blocks(out)
        out = self.conv_output(out)
        return out


# =============================
# Load Model and Weights
# =============================

def load_model(weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = SRModel(num_residuals=8, upscale_factor=4).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# =============================
# Run Inference on One Image
# =============================

def super_resolve(model, image_path, output_path):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        sr_tensor = model(input_tensor).clamp(0, 1)
        sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
        sr_image.save(output_path)
        print(f"Saved SR image to {output_path}")

# =============================
# Example Usage
# =============================

if __name__ == "__main__":
    
    model = load_model("sr_model_latest.pth")
    super_resolve(model, "dataset_lr/test/images/20230510_130236-YUGENTHRA-NAIDU-A-L-SENTHIVELL_jpg.rf.d8b7a6c8c6966f3e4d8fdad76553ec73.jpg", "sr_output.jpg")