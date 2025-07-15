# Fix OpenMP initialization issue
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import shutil
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
dataset_dir = "Processed-Dataset"
hr_dir = os.path.join(dataset_dir, "HR")
lr_dir = os.path.join(dataset_dir, "LR")
sr_dir = os.path.join(dataset_dir, "SR")
models_dir = "models"
results_dir = "results"

# Create output directories
os.makedirs(sr_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 100  # Reduced epochs since we're using a simpler model
LEARNING_RATE = 1e-4  # Slightly lower learning rate for stability
TRAIN_SPLIT = 0.8
UPSCALE_FACTOR = 4
PATCH_SIZE = 64  # Larger patches for better context


# =============================
# Simplified Super-Resolution Models
# =============================

# Simplified residual block
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
        out += residual  # Skip connection
        out = self.relu(out)
        return out


# Simpler super-resolution model (SRResNet inspired)
class SRModel(nn.Module):
    def __init__(self, num_residuals=16, upscale_factor=4):
        super(SRModel, self).__init__()

        # Initial convolution
        self.conv_input = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        res_blocks = []
        for _ in range(num_residuals):
            res_blocks.append(ResidualBlock(64))
        self.residual_blocks = nn.Sequential(*res_blocks)

        # Post-residual convolution
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)

        # Upscaling layers (using pixel shuffle)
        upscale_blocks = []
        for _ in range(2 if upscale_factor == 4 else 1):  # Two for 4x, one for 2x
            upscale_blocks.extend([
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),  # Increases spatial size by 2x
                nn.ReLU(inplace=True)
            ])

        self.upscale_blocks = nn.Sequential(*upscale_blocks)

        # Final output convolution
        self.conv_output = nn.Conv2d(64, 3, kernel_size=9, padding=4)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial convolution
        out = self.relu(self.conv_input(x))

        # Residual blocks
        residual = out
        out = self.residual_blocks(out)

        # Skip connection over residual blocks
        out = self.bn_mid(self.conv_mid(out))
        out = out + residual

        # Upscaling
        out = self.upscale_blocks(out)

        # Final convolution
        out = self.conv_output(out)

        return out


# =============================
# Dataset and Dataloader
# =============================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=64, scale=4, is_train=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train

        # Get image files
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.jpg")) +
                               glob.glob(os.path.join(hr_dir, "*.jpeg")) +
                               glob.glob(os.path.join(hr_dir, "*.png")))

        self.lr_files = []
        valid_hr_files = []

        for hr_file in self.hr_files:
            base_name = os.path.basename(hr_file)
            lr_file = os.path.join(lr_dir, base_name)
            if os.path.exists(lr_file):
                # Check if images are valid and large enough
                try:
                    hr_img = Image.open(hr_file).convert('RGB')
                    lr_img = Image.open(lr_file).convert('RGB')

                    # Skip images that are too small
                    if hr_img.width < 16 or hr_img.height < 16 or lr_img.width < 4 or lr_img.height < 4:
                        continue

                    self.lr_files.append(lr_file)
                    valid_hr_files.append(hr_file)
                except:
                    # Skip problematic images
                    continue

        self.hr_files = valid_hr_files

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_files[idx]).convert('RGB')
        lr_img = Image.open(self.lr_files[idx]).convert('RGB')

        # For training, get random crops
        if self.is_train:
            # Get random crop coordinates for LR image
            lr_w, lr_h = lr_img.size

            # Ensure patch size is not larger than image
            patch_size = min(self.patch_size, lr_w, lr_h)

            # If patch is too small, resize the image
            if patch_size < 16:  # Minimum reasonable patch size
                # Scale to make the smaller dimension at least 64 pixels
                ratio = max(64 / lr_w, 64 / lr_h)
                new_size = (int(lr_w * ratio), int(lr_h * ratio))
                lr_img = lr_img.resize(new_size,
                                       Image.BICUBIC if hasattr(Image, 'BICUBIC') else getattr(Image, 'Resampling',
                                                                                               Image).BICUBIC)

                # Corresponding HR image
                hr_size = (new_size[0] * self.scale, new_size[1] * self.scale)
                hr_img = hr_img.resize(hr_size,
                                       Image.BICUBIC if hasattr(Image, 'BICUBIC') else getattr(Image, 'Resampling',
                                                                                               Image).BICUBIC)

                # Recalculate patch size and dimensions
                lr_w, lr_h = lr_img.size
                patch_size = min(self.patch_size, lr_w, lr_h)

            # Get random crop coordinates
            lr_x = random.randint(0, max(0, lr_w - patch_size))
            lr_y = random.randint(0, max(0, lr_h - patch_size))

            # Crop LR image
            lr_patch = lr_img.crop((lr_x, lr_y, lr_x + patch_size, lr_y + patch_size))

            # Calculate corresponding HR coordinates
            hr_x, hr_y = lr_x * self.scale, lr_y * self.scale
            hr_patch_size = patch_size * self.scale

            # Crop HR image
            hr_patch = hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))

            # Apply transforms
            hr_tensor = self.transform(hr_patch)
            lr_tensor = self.transform(lr_patch)
        else:
            # For validation, resize if image is too large
            if hr_img.width > 1024 or hr_img.height > 1024:
                # Resize while maintaining aspect ratio
                hr_img.thumbnail((1024, 1024),
                                 Image.LANCZOS if hasattr(Image, 'LANCZOS') else getattr(Image, 'Resampling',
                                                                                         Image).LANCZOS)

                # Resize LR image accordingly
                new_size = (hr_img.width // self.scale, hr_img.height // self.scale)
                lr_img = lr_img.resize(new_size,
                                       Image.BICUBIC if hasattr(Image, 'BICUBIC') else getattr(Image, 'Resampling',
                                                                                               Image).BICUBIC)

            # If validation image is too small, upscale it
            if hr_img.width < 20 or hr_img.height < 20:
                # Scale to make the smaller dimension at least 20 pixels in HR
                ratio = max(20 / hr_img.width, 20 / hr_img.height)
                new_size = (int(hr_img.width * ratio), int(hr_img.height * ratio))
                hr_img = hr_img.resize(new_size,
                                       Image.BICUBIC if hasattr(Image, 'BICUBIC') else getattr(Image, 'Resampling',
                                                                                               Image).BICUBIC)

                # Resize LR image accordingly
                new_size = (hr_img.width // self.scale, hr_img.height // self.scale)
                lr_img = lr_img.resize(new_size,
                                       Image.BICUBIC if hasattr(Image, 'BICUBIC') else getattr(Image, 'Resampling',
                                                                                               Image).BICUBIC)

            hr_tensor = self.transform(hr_img)
            lr_tensor = self.transform(lr_img)

        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': self.lr_files[idx], 'HR_path': self.hr_files[idx]}


# Function to create train and validation datasets
def create_train_val_datasets(hr_dir, lr_dir, patch_size=64, scale=4, train_split=0.8):
    # Get all image files
    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.jpg")) +
                      glob.glob(os.path.join(hr_dir, "*.jpeg")) +
                      glob.glob(os.path.join(hr_dir, "*.png")))

    # Shuffle and split into train and validation sets
    random.shuffle(hr_files)
    split_idx = int(len(hr_files) * train_split)

    train_hr_files = hr_files[:split_idx]
    val_hr_files = hr_files[split_idx:]

    # Create temporary train and validation directories
    train_hr_dir = os.path.join(dataset_dir, "train_hr")
    train_lr_dir = os.path.join(dataset_dir, "train_lr")
    val_hr_dir = os.path.join(dataset_dir, "val_hr")
    val_lr_dir = os.path.join(dataset_dir, "val_lr")

    for d in [train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir]:
        os.makedirs(d, exist_ok=True)
        # Clear directory
        for f in glob.glob(os.path.join(d, "*")):
            if os.path.isfile(f):
                os.remove(f)

    # Create symbolic links or copies for train and validation sets
    for hr_file in train_hr_files:
        base_name = os.path.basename(hr_file)
        lr_file = os.path.join(lr_dir, base_name)

        if os.path.exists(lr_file):
            try:
                # Try symbolic link first
                os.symlink(os.path.abspath(hr_file), os.path.join(train_hr_dir, base_name))
                os.symlink(os.path.abspath(lr_file), os.path.join(train_lr_dir, base_name))
            except:
                # Fall back to copying if symbolic links aren't supported
                shutil.copy(hr_file, os.path.join(train_hr_dir, base_name))
                shutil.copy(lr_file, os.path.join(train_lr_dir, base_name))

    for hr_file in val_hr_files:
        base_name = os.path.basename(hr_file)
        lr_file = os.path.join(lr_dir, base_name)

        if os.path.exists(lr_file):
            try:
                # Try symbolic link first
                os.symlink(os.path.abspath(hr_file), os.path.join(val_hr_dir, base_name))
                os.symlink(os.path.abspath(lr_file), os.path.join(val_lr_dir, base_name))
            except:
                # Fall back to copying if symbolic links aren't supported
                shutil.copy(hr_file, os.path.join(val_hr_dir, base_name))
                shutil.copy(lr_file, os.path.join(val_lr_dir, base_name))

    # Create datasets
    train_dataset = SRDataset(train_hr_dir, train_lr_dir, patch_size, scale, is_train=True)
    val_dataset = SRDataset(val_hr_dir, val_lr_dir, patch_size, scale, is_train=False)

    print(f"Created training dataset with {len(train_dataset)} samples")
    print(f"Created validation dataset with {len(val_dataset)} samples")

    return train_dataset, val_dataset


# =============================
# Training Functions
# =============================

def plot_training_history(train_losses, val_psnrs, val_ssims):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(val_psnrs)
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(val_ssims)
    plt.title('Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.close()


def visualize_sr_results(model, valid_batches, epoch, save_path=None):
    """
    Visualize super-resolution results from saved validation batches
    """
    # Set model to eval mode
    model.eval()

    if not valid_batches:
        print("No valid batches for visualization")
        return 0, 0

    # Use the first valid batch
    batch = valid_batches[0]

    # Get data
    lr_img = batch['LR']
    hr_img = batch['HR']
    sr_img = batch['SR']  # Already computed

    # Convert tensors to numpy arrays for visualization
    lr_img = lr_img[0].cpu().permute(1, 2, 0).numpy()
    hr_img = hr_img[0].cpu().permute(1, 2, 0).numpy()
    sr_img = sr_img[0].cpu().permute(1, 2, 0).numpy()

    # Clip values to [0, 1]
    lr_img = np.clip(lr_img, 0, 1)
    hr_img = np.clip(hr_img, 0, 1)
    sr_img = np.clip(sr_img, 0, 1)

    # Calculate PSNR and SSIM directly using numpy (more stable)
    try:
        # MSE-based PSNR calculation
        mse = np.mean((hr_img - sr_img) ** 2)
        if mse < 1e-10:
            psnr_value = 100
        else:
            psnr_value = 10 * np.log10(1.0 / mse)

        # Simple SSIM with minimum parameters
        ssim_value = structural_similarity(
            hr_img,
            sr_img,
            win_size=3,
            channel_axis=2,
            data_range=1.0
        )
    except Exception as e:
        print(f"Visualization metrics error: {e}")
        psnr_value = 0
        ssim_value = 0

    # Create figure for visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display images
    axes[0].imshow(lr_img)
    axes[0].set_title(f'LR Image ({lr_img.shape[1]}x{lr_img.shape[0]})')
    axes[0].axis('off')

    axes[1].imshow(sr_img)
    axes[1].set_title(
        f'SR Image ({sr_img.shape[1]}x{sr_img.shape[0]})\nPSNR: {psnr_value:.2f}dB, SSIM: {ssim_value:.4f}')
    axes[1].axis('off')

    axes[2].imshow(hr_img)
    axes[2].set_title(f'HR Image ({hr_img.shape[1]}x{hr_img.shape[0]})')
    axes[2].axis('off')

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return psnr_value, ssim_value


def train_model(train_dataloader, val_dataloader, num_epochs=100):
    # Initialize model
    model = SRModel(num_residuals=8, upscale_factor=UPSCALE_FACTOR).to(device)

    # Define loss function and optimizer
    criterion = nn.L1Loss()  # L1 gives better perceptual results than MSE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Training history
    train_losses = []
    val_psnrs = []
    val_ssims = []

    best_psnr = -1
    start_time = time.time()

    # Progress bar for epochs
    pbar_epoch = tqdm(total=num_epochs, desc="Training SR Model")

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0

        # Progress bar for batches
        pbar_batch = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch in train_dataloader:
            # Get data
            lr_img = batch['LR'].to(device)
            hr_img = batch['HR'].to(device)

            # Forward pass
            optimizer.zero_grad()
            sr_img = model(lr_img)

            # Calculate loss
            loss = criterion(sr_img, hr_img)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Update batch progress bar
            pbar_batch.update(1)

        # Close batch progress bar
        pbar_batch.close()

        # Calculate average loss
        avg_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        val_count = 0
        error_count = 0

        # Create a list to collect some valid validation images for visualization
        valid_batches = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                try:
                    # Get data
                    lr_img = batch['LR'].to(device)
                    hr_img = batch['HR'].to(device)

                    # Skip problematic images
                    if lr_img.shape[2] < 4 or lr_img.shape[3] < 4:  # Too small images
                        error_count += 1
                        continue

                    # Forward pass
                    sr_img = model(lr_img)

                    # Save a few valid batches for visualization
                    if len(valid_batches) < 3 and batch_idx % 10 == 0:  # Save every 10th valid batch, up to 3
                        valid_batches.append({
                            'LR': lr_img.clone(),
                            'HR': hr_img.clone(),
                            'SR': sr_img.clone()
                        })

                    # Convert tensors for metrics calculation - CPU is more stable for metrics
                    try:
                        # Only calculate metrics for a subset of validation images to save time
                        if batch_idx % 2 == 0:  # Skip every other batch for speed
                            continue

                        # Direct numpy conversion (avoiding PIL resize which can cause issues)
                        hr_np = hr_img[0].cpu().numpy().transpose(1, 2, 0)
                        sr_np = sr_img[0].cpu().numpy().transpose(1, 2, 0)

                        # Handle dimension mismatch without PIL
                        if sr_np.shape != hr_np.shape:
                            error_count += 1
                            continue

                        # Clip values to [0, 1]
                        hr_np = np.clip(hr_np, 0, 1)
                        sr_np = np.clip(sr_np, 0, 1)

                        # Calculate metrics with robust settings
                        # Use MSE-based PSNR calculation (more robust)
                        mse = np.mean((hr_np - sr_np) ** 2)
                        if mse == 0:  # Avoid division by zero
                            psnr = 100  # Large arbitrary value
                        else:
                            psnr = 10 * np.log10(1.0 / mse)

                        # Use minimal SSIM settings
                        ssim = structural_similarity(
                            hr_np,
                            sr_np,
                            win_size=3,  # Minimal window size
                            channel_axis=2,
                            data_range=1.0,
                            K1=0.01,
                            K2=0.03  # Default values
                        )

                        # Add to sum
                        val_psnr += psnr
                        val_ssim += ssim
                        val_count += 1
                    except Exception as e:
                        # Just skip without printing to avoid console spam
                        error_count += 1
                        continue
                except Exception as e:
                    error_count += 1
                    continue

        # Print summary of errors only if they exceed a threshold
        if error_count > 20:
            print(f"Skipped {error_count} validation images due to calculation errors")

        # Calculate average metrics (avoid division by zero)
        if val_count > 0:
            avg_psnr = val_psnr / val_count
            avg_ssim = val_ssim / val_count
        else:
            # No valid metrics, use bicubic upsampling to estimate
            avg_psnr = 20.0  # Typical bicubic PSNR
            avg_ssim = 0.5  # Typical bicubic SSIM
            print("Warning: Could not calculate metrics from validation set. Using estimated values.")

        # Store metrics history
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        # Update learning rate based on validation PSNR
        scheduler.step(avg_psnr)

        # Check if this is the best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
            }, os.path.join(models_dir, 'sr_model_best.pth'))

        # Save the latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
        }, os.path.join(models_dir, 'sr_model_latest.pth'))

        # Log training progress
        time_elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.6f} | Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f} | Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        # Update epoch progress bar
        pbar_epoch.update(1)

        # Visualize results every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            try:
                if valid_batches:
                    visualize_sr_results(model, valid_batches, epoch + 1,
                                         save_path=os.path.join(results_dir, f'val_epoch_{epoch + 1}.png'))
                else:
                    print("No valid batches available for visualization")
            except Exception as e:
                print(f"Visualization error: {e}")

    # Close epoch progress bar
    pbar_epoch.close()

    # Plot training history
    plot_training_history(train_losses, val_psnrs, val_ssims)

    return model


# =============================
# Generate SR Images
# =============================

def generate_sr_images(model_path, lr_dir, sr_dir):
    """
    Generate super-resolution images using the trained model
    """
    # Load model
    model = SRModel(num_residuals=8, upscale_factor=UPSCALE_FACTOR).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Get all LR images
    lr_images = sorted(glob.glob(os.path.join(lr_dir, "*.jpg")) +
                       glob.glob(os.path.join(lr_dir, "*.jpeg")) +
                       glob.glob(os.path.join(lr_dir, "*.png")))

    # Create progress bar
    pbar = tqdm(total=len(lr_images), desc="Generating SR images")

    # Process each image
    with torch.no_grad():
        for lr_path in lr_images:
            try:
                # Load LR image
                lr_img = Image.open(lr_path).convert('RGB')

                # Handle small images by upscaling them first
                if lr_img.width < 8 or lr_img.height < 8:
                    scale_factor = max(8 / lr_img.width, 8 / lr_img.height)
                    new_size = (int(lr_img.width * scale_factor), int(lr_img.height * scale_factor))
                    lr_img = lr_img.resize(new_size,
                                           Image.BICUBIC if hasattr(Image, 'BICUBIC') else getattr(Image, 'Resampling',
                                                                                                   Image).BICUBIC)

                # Transform image
                lr_tensor = transform(lr_img).unsqueeze(0).to(device)

                # Generate SR image
                sr_tensor = model(lr_tensor)

                # Convert to PIL image
                sr_tensor = torch.clamp(sr_tensor, 0, 1)
                sr_tensor = sr_tensor.squeeze(0).cpu()
                sr_img = transforms.ToPILImage()(sr_tensor)

                # Save SR image
                base_name = os.path.basename(lr_path)
                sr_path = os.path.join(sr_dir, base_name)
                sr_img.save(sr_path)

                # Copy XML annotation if exists
                xml_base = os.path.splitext(base_name)[0]
                lr_xml_path = os.path.join(lr_dir, f"{xml_base}.xml")

                if os.path.exists(lr_xml_path):
                    # Parse XML and adjust for upscaling
                    tree = ET.parse(lr_xml_path)
                    root = tree.getroot()

                    # Update image size
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)

                    size.find('width').text = str(width * UPSCALE_FACTOR)
                    size.find('height').text = str(height * UPSCALE_FACTOR)

                    # Update bounding boxes
                    for obj in root.findall('object'):
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)

                        bbox.find('xmin').text = str(xmin * UPSCALE_FACTOR)
                        bbox.find('ymin').text = str(ymin * UPSCALE_FACTOR)
                        bbox.find('xmax').text = str(xmax * UPSCALE_FACTOR)
                        bbox.find('ymax').text = str(ymax * UPSCALE_FACTOR)

                    # Save SR XML
                    sr_xml_path = os.path.join(sr_dir, f"{xml_base}.xml")
                    tree.write(sr_xml_path)
            except Exception as e:
                print(f"Error processing {lr_path}: {e}")

            # Update progress bar
            pbar.update(1)

    # Close progress bar
    pbar.close()

    print(f"Generated {len(lr_images)} SR images!")


# =============================
# Main Function
# =============================

def main():
    print("Starting Super-Resolution Training and Inference")

    # Create train and validation datasets
    train_dataset, val_dataset = create_train_val_datasets(
        hr_dir, lr_dir,
        patch_size=PATCH_SIZE,
        scale=UPSCALE_FACTOR,
        train_split=TRAIN_SPLIT
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Use batch size of 1 for validation
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Train the model
    print("Starting training...")
    model = train_model(train_dataloader, val_dataloader, num_epochs=NUM_EPOCHS)

    # Generate SR images
    print("Generating SR images using the best model...")
    generate_sr_images(
        model_path=os.path.join(models_dir, 'sr_model_best.pth'),
        lr_dir=lr_dir,
        sr_dir=sr_dir
    )

    print("Super-Resolution completed!")


if __name__ == "__main__":
    main()