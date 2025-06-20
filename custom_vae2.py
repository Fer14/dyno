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
    def __init__(self, latent_dim=32):  # Start with intermediate latent size
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_var = nn.Linear(256 * 16 * 16, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 16 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 16, 16)
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
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
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


def train_improved_vae(
    fix_latent_space,
    data_folder,
    num_epochs=200,
    batch_size=8,  # Increased for 256x256
    learning_rate=1e-4,
    beta_schedule=None,
    save_dir="improved_vae_checkpoints",
    latent_dim=128,  # Much larger latent space
    reconstruction_weight=100.0,  # Higher weight for better color preservation
):
    """
    Train Improved VAE with enhanced features for dragon generation
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Enhanced data transforms for 256x256 with color augmentation
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Changed to 256x256
            transforms.ToTensor(),
            # Using [0, 1] range instead of [-1, 1] for better color preservation with Sigmoid output
        ]
    )

    dataset = ImageDataset(data_folder, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Increased workers
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Memory optimization
    )

    # Model initialization with improved architecture
    model = VAE(latent_dim=latent_dim).to(device)

    # Uncomment to load from checkpoint
    # model = load_trained_improved_vae(
    #     checkpoint_path="/path/to/your/checkpoint.pth",
    #     latent_dim=latent_dim,
    # )

    # Enhanced optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 3,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        anneal_strategy="cos",
    )

    # Enhanced beta scheduling for better training stability
    if beta_schedule is None:
        # Gradual increase of KL weight
        warmup_epochs = num_epochs // 4
        beta_schedule = np.concatenate(
            [
                np.linspace(0.01, 0.1, warmup_epochs),  # Very gradual start
                np.linspace(0.1, 1.0, warmup_epochs),  # Gradual increase
                np.ones(num_epochs - 2 * warmup_epochs),  # Full KL weight
            ]
        )

    # Training metrics
    model.train()
    losses = []
    recon_losses = []
    kl_losses = []
    best_loss = float("inf")
    best_checkpoint_path = None

    print(f"Training Improved VAE on {len(dataset)} images...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Latent dimension: {latent_dim}")
    print("Image size: 256x256")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        beta = beta_schedule[min(epoch, len(beta_schedule) - 1)]

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar, z = model(data)

            # Enhanced loss computation
            loss, recon_loss, kl_loss = vae_loss(
                recon_batch,
                data,
                mu,
                logvar,
                beta=beta,
                reconstruction_weight=reconstruction_weight,
            )

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step scheduler every batch for OneCycleLR

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.2f}",
                    "Recon": f"{recon_loss.item():.2f}",
                    "KL": f"{kl_loss.item():.2f}",
                    "Beta": f"{beta:.3f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        # Calculate average losses
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
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "losses": losses,
            "recon_losses": recon_losses,
            "kl_losses": kl_losses,
            "latent_dim": latent_dim,
        }

        # Save best model
        if avg_loss < best_loss:
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)

            best_checkpoint_path = os.path.join(
                save_dir, f"improved_vae_best_epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, best_checkpoint_path)
            best_loss = avg_loss

        # Generate samples more frequently to monitor diversity
        if (epoch + 1) % 50 == 0:
            generate_diverse_samples(fix_latent_space, model, save_dir, epoch + 1)

        # Save training curves
        # if (epoch + 1) % 20 == 0:
        #     plot_training_curves(losses, recon_losses, kl_losses, save_dir, epoch + 1)

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

    if checkpoint_path.endswith(".pth") and "checkpoint" in checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
    # Enhanced Configuration
    DATA_FOLDER = "/home/fer/Escritorio/dragons/dragons/images/color"
    EPOCHS = 1000  # More epochs for better convergence best 5k
    BATCH_SIZE = 64  # Increased for 256x256 best32
    LEARNING_RATE = 1e-4
    LATENT_DIM = 512  # Much larger latent space est 64
    RECONSTRUCTION_WEIGHT = 1.0  # Higher weight for color preservation best100
    num_samples = 8

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
        data_folder=DATA_FOLDER,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        latent_dim=LATENT_DIM,
        reconstruction_weight=RECONSTRUCTION_WEIGHT,
        save_dir="improved_vae_checkpoints_3",
    )

    print("Training completed!")

    # Test diversity
    print("Testing dragon diversity...")
    test_latent_diversity(trained_model, LATENT_DIM)

    # Test interpolations
    print("Testing enhanced interpolations...")
    enhanced_interpolation(trained_model)

    print("All tests completed!")
