"""This script takes a trained VAE model and uses latent representations to rain RNN-MDN model P(z_{t+1}|z_t,a_t)
"""
import glob
import os

import numpy as np
import torch
import wandb
from rnn_mdn import MDNRNN, MDNRNNDataset, debug_shapes
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from vae import VanillaVAE


def load_vae(vae_path, configs, device):
    """Load and return the trained VAE model."""
    vae = VanillaVAE(in_channels=3, **configs).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    return vae


def encode_rollouts(vae, data_dir, device, batch_size=32, dataset_size=30):
    """Process rollouts through VAE to get latent representations."""
    transform = transforms.Compose([transforms.ToTensor()])
    latent_sequences = []
    action_sequences = []

    # Get all rollout files
    rollout_files = glob.glob(os.path.join(data_dir, "*.npz"))
    rollout_files = rollout_files[:dataset_size]

    for file in tqdm(rollout_files, desc="Processing rollouts"):
        data = np.load(file)
        observations = data["observations"]
        actions = data["actions"]

        # Process observations in batches
        latents = []
        for i in range(0, len(observations), batch_size):
            batch_obs = observations[i : i + batch_size]
            # Transform and move to device
            batch_tensor = torch.stack([transform(obs) for obs in batch_obs]).to(device)

            with torch.no_grad():
                mu, log_var = vae.encode(batch_tensor)
                # Use mean of encoder output as latent representation
                batch_latents = mu.cpu().numpy()
            latents.extend(batch_latents)

        latent_sequences.append(np.array(latents))
        action_sequences.append(actions)

    return latent_sequences, action_sequences


def create_datasets(latent_sequences, action_sequences, sequence_length=64, train_split=0.9):
    """Create train and validation datasets."""
    # Randomly split sequences into train and validation
    num_sequences = len(latent_sequences)
    indices = np.random.permutation(num_sequences)
    train_idx = indices[: int(train_split * num_sequences)]
    val_idx = indices[int(train_split * num_sequences) :]

    train_dataset = MDNRNNDataset(
        [latent_sequences[i] for i in train_idx], [action_sequences[i] for i in train_idx], sequence_length
    )

    val_dataset = MDNRNNDataset(
        [latent_sequences[i] for i in val_idx], [action_sequences[i] for i in val_idx], sequence_length
    )

    return train_dataset, val_dataset


def validate_model(model, val_loader, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            latent_seq = batch["latent"].to(device)
            action_seq = batch["action"].to(device)
            next_latent_seq = batch["next_latent"].to(device)

            pi, mu, sigma, _ = model(latent_seq, action_seq)
            loss = model.get_mixture_loss(pi, mu, sigma, next_latent_seq)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    # Configuration
    configs = {
        "vae_path": "weights/vae_50.pth",
        "data_dir": "data/dataset/",
        "latent_dim": 32,
        "hidden_dim": 256,
        "num_gaussians": 5,
        "sequence_length": 64,
        "batch_size": 32,
        "num_epochs": 20,
        "learning_rate": 1e-3,
        "action_dim": 3,  # CarRacing has 3 actions: steering, gas, brake
        "model_save_dir": "weights/",
        "gradient_clip": 1.0,
        "vae_batch_size": 32,
    }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(project="world_models_mdn_rnn", config=configs)

    # Load VAE and encode rollouts
    print("Loading VAE and encoding rollouts...")
    vae = load_vae(configs["vae_path"], {"latent_dim": configs["latent_dim"]}, device)
    latent_sequences, action_sequences = encode_rollouts(
        vae, configs["data_dir"], device, configs["vae_batch_size"], dataset_size=5
    )

    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(latent_sequences, action_sequences, configs["sequence_length"])

    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=configs["batch_size"], shuffle=False, num_workers=4)

    # Initialize MDN-RNN
    model = MDNRNN(
        latent_dim=configs["latent_dim"],
        action_dim=configs["action_dim"],
        hidden_dim=configs["hidden_dim"],
        num_gaussians=configs["num_gaussians"],
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["learning_rate"])
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    # Training loop
    for epoch in range(configs["num_epochs"]):
        model.train()
        total_train_loss = 0
        valid_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{configs['num_epochs']}")):
            latent_seq = batch["latent"].to(device)
            action_seq = batch["action"].to(device)
            next_latent_seq = batch["next_latent"].to(device)

            # Debug shapes on first batch
            if batch_idx == 0:
                print(
                    f"\nBatch shapes - Latent: {latent_seq.shape}, Action: {action_seq.shape}, Next: {next_latent_seq.shape}"
                )

            optimizer.zero_grad()
            pi, mu, sigma, _ = model(latent_seq, action_seq)
            loss = model.get_mixture_loss(pi, mu, sigma, next_latent_seq)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs["gradient_clip"])
                optimizer.step()
                total_train_loss += loss.item()
                valid_batches += 1
            else:
                print(f"Skipping batch {batch_idx} due to invalid loss")

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Compute average losses
        avg_train_loss = total_train_loss / valid_batches if valid_batches > 0 else float("inf")
        avg_val_loss = validate_model(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{configs['num_epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})

        # Save best model and implement early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Saving best model...")
            os.makedirs(configs["model_save_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(configs["model_save_dir"], "mdn_rnn_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(configs["model_save_dir"], f"mdn_rnn_{epoch+1}.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(configs["model_save_dir"], "mdn_rnn_final.pth"))
    wandb.finish()


if __name__ == "__main__":
    main()
