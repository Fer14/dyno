import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),  # 256 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 128 → 64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # 64 → 32
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, 2, 1),  # 32 → 16
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 4, 2, 1),  # 16 → 8
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, 4, 2, 1),  # 8 → 4
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(2048 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(2048 * 4 * 4, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 2048 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 2048, 4, 2, 1),  # 4 → 8
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),  # 8 → 16
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 16 → 32
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 32 → 64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 64 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 128 → 256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # keep size 256
            nn.Sigmoid(),  # tanh
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 2048, 4, 4)
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z


class ImageDataset(Dataset):
    def __init__(self, image_folders, transform=None):
        if isinstance(image_folders, str):
            image_folders = [image_folders]  # Allow single string input too

        self.image_paths = []
        for folder in image_folders:
            for f in os.listdir(folder):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(folder, f))

        self.transform = transform
        print(f"Found {len(self.image_paths)} images in {len(image_folders)} folders.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


# Enhanced loss function for better color diversity
def vae_loss(recon_x, x, mu, logvar, beta=1.0, reconstruction_weight=1.0):
    """
    Enhanced VAE loss with adjustable beta for KL divergence
    and reconstruction weight for better color preservation
    """
    # Reconstruction loss - using MSE for better color preservation
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss with beta weighting
    total_loss = reconstruction_weight * recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cyclical_beta_schedule(
    num_epochs, cycles=16, ratio=0.5, beta_start=0, beta_stop=1.0
):
    """
    Create a cyclical beta schedule that increases and resets multiple times during training.

    Args:
        num_epochs (int): Total number of training epochs.
        cycles (int): Number of cycles.
        ratio (float): Fraction of each cycle used for increasing beta.
        beta_start (float): Starting beta value (usually 0.0).
        beta_stop (float): Maximum beta value (typically 1.0).

    Returns:
        np.ndarray: A beta schedule of shape (num_epochs,).
    """
    schedule = np.zeros(num_epochs)
    period = num_epochs // cycles
    step = (beta_stop - beta_start) / (period * ratio)

    for c in range(cycles):
        for i in range(period):
            epoch = c * period + i
            if epoch >= num_epochs:
                break
            if i < period * ratio:
                schedule[epoch] = beta_start + step * i
            else:
                schedule[epoch] = beta_stop  # Flat at the top

    return schedule


def train_improved_vae(
    fix_latent_space,
    dataloader,
    num_epochs=200,
    learning_rate=1e-4,
    beta_schedule=None,
    save_dir="improved_vae_checkpoints",
    latent_dim=128,  # Much larger latent space
    reconstruction_weight=100.0,  # Higher weight for better color preservation
):
    """
    Train Improved VAE with enhanced features for dragon generation
    """
    os.makedirs(save_dir, exist_ok=True)

    model = VAE(latent_dim=latent_dim).to(device)

    # Uncomment to load from checkpoint
    model = load_trained_improved_vae(
        checkpoint_path="/home/fer/Escritorio/dragons/dragon/vae/VAE_TOTAL_4/checkpoint_epoch_181_interrupted.pth",
        latent_dim=latent_dim,
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    ######
    # warmup_epochs = num_epochs // 4
    # beta_schedule = np.concatenate(
    #     [
    #         np.linspace(0.01, 0.1, warmup_epochs),  # Epochs 1–250
    #         np.linspace(0.1, 1.0, warmup_epochs),  # Epochs 251–500
    #         np.ones(500),  # Epochs 501–1000
    #     ]
    # )
    #####

    beta_schedule = build_cyclical_beta_schedule(
        num_epochs=num_epochs, cycles=4, ratio=0.5, beta_start=0.0, beta_stop=1.0
    )

    model.train()
    losses = []
    recon_losses = []
    kl_losses = []
    best_loss = float("inf")
    best_checkpoint_path = None

    print(f"Training Improved VAE on {len(dataset)} images...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Latent dimension: {latent_dim}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        beta = beta_schedule[min(epoch, len(beta_schedule) - 1)]

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, data in enumerate(progress_bar):
            try:
                data = data.to(device, non_blocking=True)

                optimizer.zero_grad()

                recon_batch, mu, logvar, z = model(data)

                loss, recon_loss, kl_loss = vae_loss(
                    recon_batch,
                    data,
                    mu,
                    logvar,
                    beta=beta,
                    reconstruction_weight=reconstruction_weight,
                )

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.2f}",
                        "Recon": f"{recon_loss.item():.2f}",
                        "KL": f"{kl_loss.item():.2f}",
                        "Beta": f"{beta:.3f}",
                    }
                )
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt: Saving model checkpoint before exiting...")
                checkpoint_path = os.path.join(
                    save_dir, f"checkpoint_epoch_{epoch}_interrupted.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_loss / len(dataloader),
                        "recon_loss": epoch_recon_loss / len(dataloader),
                        "kl_loss": epoch_kl_loss / len(dataloader),
                        "beta": beta,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved to {checkpoint_path}")
                break  # Exit the training loop

        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)

        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        print(
            f"Epoch {epoch + 1}: Loss={avg_loss:.2f}, Recon={avg_recon:.2f}, KL={avg_kl:.2f}, Beta={beta:.3f}"
        )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "losses": losses,
            "recon_losses": recon_losses,
            "kl_losses": kl_losses,
            "latent_dim": latent_dim,
        }

        # Save best model
        if (avg_recon + avg_kl) < best_loss:
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)

            best_checkpoint_path = os.path.join(
                save_dir, f"improved_vae_best_epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, best_checkpoint_path)
            best_loss = avg_recon + avg_kl

        # Generate samples more frequently to monitor diversity
        if (epoch + 1) % 10 == 0:
            generate_diverse_samples(fix_latent_space, model, save_dir, epoch + 1)

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "improved_vae_final.pth"))

    return model, losses


def generate_diverse_samples(z, model, save_dir, epoch):
    """Generate diverse sample images to check color and feature variation."""
    model.eval()
    with torch.no_grad():
        # Sample from different regions of latent space for diversity

        samples = model.decode(z)

        # Create grid with larger images
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(3 * num_samples // 2, 6))
        axes = axes.flatten() if num_samples > 2 else [axes]

        for i, ax in enumerate(axes):
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Sample {i + 1}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"diverse_samples_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    model.train()


def plot_training_curves(losses, recon_losses, kl_losses, save_dir, epoch):
    """Plot training curves to monitor progress."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(losses)
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(recon_losses)
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)

    axes[2].plot(kl_losses)
    axes[2].set_title("KL Divergence Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"training_curves_epoch_{epoch}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def load_trained_improved_vae(checkpoint_path, latent_dim=128):
    """Load a trained Improved VAE model"""
    model = VAE(latent_dim=latent_dim)

    if checkpoint_path.endswith(".pth"):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return model.to(device)


def test_latent_diversity(model, latent_dim=128, num_tests=16):
    """Test the diversity of generated samples."""
    model.eval()
    with torch.no_grad():
        # Generate samples with different sampling strategies
        z_normal = torch.randn(num_tests // 2, latent_dim, device=device)
        z_scaled = torch.randn(num_tests // 2, latent_dim, device=device) * 1.5

        z_combined = torch.cat([z_normal, z_scaled], dim=0)
        samples = model.decode(z_combined)

        # Create a large grid to visualize diversity
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Dragon {i + 1}")

        plt.suptitle("Generated Dragon Diversity Test", fontsize=16)
        plt.tight_layout()
        plt.savefig("dragon_diversity_test.png", dpi=150, bbox_inches="tight")
        plt.show()

    model.train()


def enhanced_interpolation(model, num_interpolations=10, steps=8):
    """Enhanced interpolation between multiple random points."""
    model.eval()

    fig, axes = plt.subplots(
        num_interpolations, steps, figsize=(2 * steps, 2 * num_interpolations)
    )

    with torch.no_grad():
        for row in range(num_interpolations):
            # Generate two random points
            z1 = torch.randn(model.latent_dim, device=device)
            z2 = torch.randn(model.latent_dim, device=device)

            # Interpolate
            for col, alpha in enumerate(np.linspace(0, 1, steps)):
                z_interp = (1 - alpha) * z1 + alpha * z2
                img = model.decoder(z_interp.unsqueeze(0))
                img = img.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)

                axes[row, col].imshow(img)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"α={alpha:.2f}")

    plt.suptitle("Enhanced Latent Space Interpolations", fontsize=16)
    plt.tight_layout()
    plt.savefig("enhanced_interpolations.png", dpi=150, bbox_inches="tight")
    plt.show()

    model.train()


# Example usage
if __name__ == "__main__":
    EPOCHS = 1000  # More epochs for better convergence best 5k
    BATCH_SIZE = 32  # Increased for 256x256 best32
    LEARNING_RATE = 1e-5
    LATENT_DIM = 1024  # Much larger latent space est 64
    RECONSTRUCTION_WEIGHT = 1.0  # Higher weight for color preservation best100
    num_samples = 8
    IMG_SIZE = 256  # Increased image size for better quality

    # Enhanced Configuration
    data_folders = [
        "/home/fer/Escritorio/dragons/dragon/dragons/images/color",
        "/home/fer/Descargas/dragones_lambda/dragons/images",
        "/home/fer/Descargas/dragones_lambda2/dragons/images",
        "/home/fer/Descargas/dragones_lambda3/images",
        "/home/fer/Descargas/dragones_lambda4/dragons/images",
        "/home/fer/Descargas/dragones_lambda5/images",
        "/home/fer/Descargas/dragones_lambda6/dragons/images",
        "/home/fer/Descargas/dragones_lambda7/dragons/images",
        "/home/fer/Descargas/dragones_lambda8/dragons/images",
        "/home/fer/Descargas/dragones_lambda9/dragons/images",
        "/home/fer/Descargas/dragones_lambda10/dragons/images",
        "/home/fer/Descargas/dragones_lambda11/images",
    ]

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = ImageDataset(data_folders, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Memory optimization settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Enable mixed precision for memory efficiency (optional)
    # torch.backends.cuda.matmul.allow_tf32 = True

    # Start training
    print("Starting Improved VAE training...")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    fix_latent_space = torch.randn(num_samples, LATENT_DIM, device=device)

    # Add some structured sampling for better diversity visualization
    if num_samples >= 4:
        # Some samples from different scales
        fix_latent_space[: num_samples // 2] *= 0.8  # Closer to mean
        fix_latent_space[num_samples // 2 :] *= 1.5  # Further from mean

    # Train the model
    trained_model, training_losses = train_improved_vae(
        fix_latent_space=fix_latent_space,
        dataloader=dataloader,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        latent_dim=LATENT_DIM,
        reconstruction_weight=RECONSTRUCTION_WEIGHT,
        save_dir="VAE_TOTAL_4",
    )

    print("Training completed!")

    # Test diversity
    print("Testing dragon diversity...")
    test_latent_diversity(trained_model, LATENT_DIM)

    # Test interpolations
    print("Testing enhanced interpolations...")
    enhanced_interpolation(trained_model)

    print("All tests completed!")


####

# https://chatgpt.com/c/688ba4b7-c884-8328-9b78-8998ddde3d3b

# Add residual connections (ResNet-style VAE)

# Use attention (like VQ-VAE-2)

# Add hierarchical latent structure (e.g. Ladder VAE)
