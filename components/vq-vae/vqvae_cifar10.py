"""VQ-VAE on CIFAR-10 dataset with wandb integration and lr scheduler"""
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
from box import Box
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class Config:
    def __init__(self):
        self.project_name = "vqvae-cifar10"
        self.seed = 42
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Dataset parameters
        self.data_dir = "./data"
        self.dataset = "cifar10"
        self.image_size = 32
        self.channels = 3
        self.batch_size = 32 # 128 prev

        # Model parameters
        self.latent_dim = 256
        self.num_embeddings = 128 # Size of the codebook (K)
        self.commitment_cost = 1  # Beta in paper

        # Training parameters
        self.learning_rate =  3e-4
        self.num_epochs = 200
        self.log_interval = 100
        self.save_interval = 10
        self.lr_patience = 30 # Number of epochs to wait before reducing learning rate
        self.lr_factor = 0.4  # Factor by which to reduce learning rate

        # Paths
        self.save_dir = Path(f"./models/{wandb.run.name if wandb.run else 'default_run'}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def seed_everything(seed: int):
    """Set all random seeds for reproducibility"""
    print(f"Seeding everything with {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def show_image(batch_of_tensors, title=None, wandb_log=False, step=None):
    """Show a batch of images with option to log to wandb"""
    images = batch_of_tensors[:4]
    images = (images * 0.5) + 0.5  # Unnormalize the images to [0, 1] range
    grid_img = torchvision.utils.make_grid(images, nrow=2)

    if not wandb_log:
        plt.figure(figsize=(5, 5))
        plt.imshow(grid_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        images = wandb.Image(grid_img.permute(1, 2, 0).numpy(), caption=title)
        wandb.log({title: images}, step=step)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE

    Attributes:
        num_embeddings: Size of the codebook (K in the paper)
        embedding_dim: Dimension of each embedding vector (D in the paper)
        commitment_cost: Beta parameter controlling commitment loss
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Init embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        """
        Args:
            z: Encoder output tensor [B, D, H, W]

        Returns:
            z_q: Quantized tensor [B, D, H, W]
            loss: VQ loss
            encoding_indices: Indices of the nearest embeddings [B*H*W]
        """
        b, c, h, w = z.shape
        assert c == self.embedding_dim, f"Input channels {c} must match embedding dimension {self.embedding_dim}"
        assert h == w, f"Input height {h} must match width {w}"
        # Change to [B, H, W, D] for easier processing
        z_channel_last = z.permute(0, 2, 3, 1)
        z_flattened = z_channel_last.reshape(-1, self.embedding_dim)

        # Calculate distances between z and the codebook embeddings |a-b|²
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)  # a²
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()  # b²
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())  # -2ab
        )

        # Get the index with the smallest distance
        encoding_indices = torch.argmin(distances, dim=1)

        # Get the quantized vector
        z_q = self.embedding(encoding_indices)
        z_q = z_q.view(b, h, w, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)  # [B, D, H, W]

        # Calculate the commitment loss
        # MSE between quantized vectors and encoder outputs
        q_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        # MSE between quantized vectors and encoder outputs (with gradients for codebook)
        e_latent_loss = torch.mean((z_q - z.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator trick for gradient backpropagation
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


class ResidualBlock(nn.Module):
    """
    Residual block for VQVAE
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual  # Skip connection
        x = self.relu(x)
        return x

class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder with Residual Connections

    Attributes:
        encoder: Encoder network
        vq_layer: Vector quantization layer
        decoder: Decoder network
    """

    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.config = config
        
        # Number of residual blocks per layer
        num_res_blocks = 2
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            # Initial convolution
            nn.Sequential(
                nn.Conv2d(config.channels, 32, kernel_size=4, stride=2, padding=1),  # 32x32x3 -> 16x16x32
                nn.ReLU()
            ),
            # First residual stage
            nn.Sequential(
                *[ResidualBlock(32) for _ in range(num_res_blocks)],
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16x32 -> 8x8x64
                nn.ReLU()
            ),
            # Second residual stage
            nn.Sequential(
                *[ResidualBlock(64) for _ in range(num_res_blocks)],
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8x64 -> 4x4x128
                nn.ReLU()
            ),
            # Final convolution to latent space
            nn.Sequential(
                *[ResidualBlock(128) for _ in range(num_res_blocks)],
                nn.Conv2d(128, config.latent_dim, kernel_size=1)  # 4x4x128 -> 4x4xlatent_dim
            )
        ])
                
        # Vector Quantization
        self.vq_layer = VectorQuantizer(
            config.num_embeddings, config.latent_dim, commitment_cost=config.commitment_cost
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            # Initial convolution from latent space
            nn.Sequential(
                nn.ConvTranspose2d(config.latent_dim, 128, kernel_size=1),  # 4x4xlatent_dim -> 4x4x128
                nn.ReLU(),
                *[ResidualBlock(128) for _ in range(num_res_blocks)]
            ),
            # First upsampling stage
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4x128 -> 8x8x64
                nn.ReLU(),
                *[ResidualBlock(64) for _ in range(num_res_blocks)]
            ),
            # Second upsampling stage
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8x64 -> 16x16x32
                nn.ReLU(),
                *[ResidualBlock(32) for _ in range(num_res_blocks)]
            ),
            # Final convolution to output
            nn.Sequential(
                nn.ConvTranspose2d(32, config.channels, kernel_size=4, stride=2, padding=1),  # 16x16x32 -> 32x32x3
                nn.Tanh()  # Output values in range [-1, 1]
            )
        ])

    def encoder(self, x):
        """Forward pass through encoder with residual blocks"""
        for layer in self.encoder_layers:
            x = layer(x)
        return x
        
    def decoder(self, x):
        """Forward pass through decoder with residual blocks"""
        for layer in self.decoder_layers:
            x = layer(x)
        return x

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            x_recon: Reconstructed tensor [B, C, H, W]
            vq_loss: Vector quantization loss
        """
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices

    def encode(self, x):
        """Encode input to quantized representation"""
        z_e = self.encoder(x)
        z_q, _, indices = self.vq_layer(z_e)
        return z_q, indices

    def decode(self, z_q):
        """Decode from quantized representation"""
        return self.decoder(z_q)

    def save(self, path):
        """Save model state"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model state with support for different devices"""
        # Load weights with device mapping to CPU first, then move to target device if needed
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(state_dict)
        self.to(self.config.device)  # Move to the appropriate device
        print(f"Model loaded from {path} and moved to {self.config.device}")

        

def vqvae_loss(recon_x, x, vq_loss):
    """Calculate VQ-VAE loss"""
    recon_loss = F.mse_loss(recon_x, x)
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss


def get_data_loaders(config):
    """Create data loaders for training and testing"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    if config.dataset.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root=config.data_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=config.data_dir, train=False, transform=transform, download=True)
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not implemented")

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def evaluate(model, test_loader, config, global_step):
    """Evaluate model on test set"""
    model.eval()
    total_test_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.device)
            recon_batch, vq_loss, _ = model(data)
            loss, recon_loss = vqvae_loss(recon_batch, data, vq_loss)

            total_test_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

    # Average losses
    avg_test_loss = total_test_loss / len(test_loader)
    avg_recon_loss = total_recon_loss / len(test_loader)
    avg_vq_loss = total_vq_loss / len(test_loader)

    # Log to wandb
    wandb.log(
        {"test/loss": avg_test_loss, "test/recon_loss": avg_recon_loss, "test/vq_loss": avg_vq_loss}, step=global_step
    )

    # Visualize reconstructions
    with torch.no_grad():
        data = next(iter(test_loader))[0][:4].to(config.device)
        recon, _, _ = model(data)
        show_image(data.cpu(), "Test Original", wandb_log=True, step=global_step)
        show_image(recon.cpu(), "Test Reconstruction", wandb_log=True, step=global_step)

    return avg_test_loss


def train(model, train_loader, test_loader, optimizer, scheduler, config):
    """Train the model"""
    global_step = 0
    best_loss = float("inf")

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_vq_loss = 0

        # Use tqdm for the progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for batch_idx, (data, _) in pbar:
            data = data.to(config.device)
            global_step += 1

            optimizer.zero_grad()
            recon_batch, vq_loss, _ = model(data)
            loss, recon_loss = vqvae_loss(recon_batch, data, vq_loss)
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f'{optimizer.param_groups[0]["lr"]:.6f}'})

            # Log to wandb
            if batch_idx % config.log_interval == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/vq_loss": vq_loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        # Average losses for the epoch
        avg_loss = train_loss / len(train_loader)
        avg_recon_loss = train_recon_loss / len(train_loader)
        avg_vq_loss = train_vq_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] "
            f"Average Loss: {avg_loss:.4f}, "
            f"Recon Loss: {avg_recon_loss:.4f}, "
            f"VQ Loss: {avg_vq_loss:.4f}"
        )

        # Evaluate on test set
        test_loss = evaluate(model, test_loader, config, global_step)

        # Step the scheduler (adjust learning rate based on validation loss)
        scheduler.step(test_loss)
        print(f"Learning rate adjusted to: {scheduler.get_last_lr()[0]:.6f}")

        # Save model if it's the best so far
        if test_loss < best_loss:
            best_loss = test_loss
            model.save(config.save_dir / f"vqvae_best.pt")
            wandb.run.summary["best_loss"] = best_loss
            print(f"New best model saved with loss: {best_loss:.4f}")

        # Save model every save_interval epochs
        if (epoch + 1) % config.save_interval == 0:
            model.save(config.save_dir / f"vqvae_epoch{epoch+1}.pt")

            # Visualize training reconstructions
            with torch.no_grad():
                recon_batch, _, _ = model(data)
                show_image(data.cpu(), "Training Original", wandb_log=True, step=global_step)
                show_image(recon_batch.cpu(), "Training Reconstruction", wandb_log=True, step=global_step)

    # Save final model
    model.save(config.save_dir / "vqvae_final.pt")
    return model


def check_model_dimensions(config):
    """
    Test function to verify dimensions are consistent throughout the model.
    Takes a sample tensor through each step of the forward pass and checks dimensions.
    """
    print("\n=== Testing model dimensions ===")

    # Create model
    model = VQVAE(config).to(config.device)
    model.eval()

    # Create sample batch
    batch_size = 2
    sample_input = torch.randn(batch_size, config.channels, config.image_size, config.image_size).to(config.device)
    print(f"Input shape: {sample_input.shape}")

    # Check encoder
    with torch.no_grad():
        # Encoder output
        z_e = model.encoder(sample_input)
        expected_z_shape = (batch_size, config.latent_dim, config.image_size // 8, config.image_size // 8)
        assert (
            z_e.shape == expected_z_shape
        ), f"Encoder output shape {z_e.shape} doesn't match expected {expected_z_shape}"
        print(f"Encoder output shape: {z_e.shape} ✓")

        # Vector quantizer
        z_q, vq_loss, indices = model.vq_layer(z_e)
        assert z_q.shape == z_e.shape, f"Quantized shape {z_q.shape} doesn't match encoder output {z_e.shape}"
        print(f"Quantized shape: {z_q.shape} ✓")
        print(f"VQ Loss: {vq_loss.item():.4f}")

        # Expected indices shape (flattened spatial dimensions)
        expected_indices_shape = (batch_size * (config.image_size // 8) * (config.image_size // 8),)
        assert (
            indices.shape == expected_indices_shape
        ), f"Indices shape {indices.shape} doesn't match expected {expected_indices_shape}"
        print(f"Indices shape: {indices.shape} ✓")

        # Decoder
        output = model.decoder(z_q)
        expected_output_shape = (batch_size, config.channels, config.image_size, config.image_size)
        assert (
            output.shape == expected_output_shape
        ), f"Output shape {output.shape} doesn't match input {sample_input.shape}"
        print(f"Decoder output shape: {output.shape} ✓")

        # Full model forward pass
        recon, vq_loss, indices = model(sample_input)
        assert (
            recon.shape == sample_input.shape
        ), f"Reconstruction shape {recon.shape} doesn't match input {sample_input.shape}"
        print(f"Full model reconstruction shape: {recon.shape} ✓")

    print("All dimension checks passed successfully!")
    return True


def main():
    """Main function to run the training pipeline"""
    config = Config()
    cfg = Box(config.to_dict())

    wandb.init(
        project=cfg.project_name,
        config=cfg.to_dict(),
        name=f"vqvae-{cfg.dataset}-{cfg.latent_dim}d-{cfg.num_embeddings}emb",
    )
    # Set the save directory for the model
    cfg.save_dir = Path(f"./models/{wandb.run.name}")
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)

    train_loader, test_loader = get_data_loaders(cfg)

    model = VQVAE(cfg).to(cfg.device)
    wandb.watch(model, log="all")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)

    print(f"Starting training on {cfg.device}")
    model = train(model, train_loader, test_loader, optimizer, scheduler, cfg)

    wandb.finish()

    return model


if __name__ == "__main__":
    config = Config()
    cfg = Box(config.to_dict())

    # Run dimension check
    check_model_dimensions(cfg)

    # Run main training
    main()


