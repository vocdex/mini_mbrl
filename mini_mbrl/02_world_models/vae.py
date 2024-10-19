"""Code is imported from https://github.com/AntixK/PyTorch-VAE/tree/master/models/
Small modifications are made to the original code: 
- When logging KL loss, the negative sign is removed.
- The layer sizes are changed to match the input size of the dataset.

"""

import torch
import numpy as np
import os
import glob
import wandb
import random
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from types_ import *
from abc import abstractmethod
from tqdm import tqdm


def seed_everything(seed: int):
    print(f"Seeding everything with {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):

    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)  # Flatten starting from dimension 1

        # print(result.shape)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VAEDatasetSingle(torch.utils.data.Dataset):
    def __init__(
        self, data_dir: str, transform=None, action_transform=None, action_return=False
    ):
        self.images = np.load(data_dir)["observations"]
        self.actions = np.load(data_dir)["actions"]
        self.transform = transform
        self.action_transform = action_transform
        self.action_return = action_return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample_image = self.images[idx]
        if self.transform:
            sample_image = self.transform(sample_image)

        if self.action_return:
            sample_action = self.actions[idx]
            if self.action_transform:
                sample_action = self.action_transform(sample_action)
            return sample_image, sample_action
        return sample_image


class VAEDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir: str, transform=None, action_transform=None, action_return=False
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.action_transform = action_transform
        self.action_return = action_return

        # Load all .npz files from the directory
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        # select only the first 50 files
        self.files = self.files[:50]
        self.observations = []
        self.actions = []

        # Combine all rollouts into one list
        for file in self.files:
            data = np.load(file)
            self.observations.extend(data["observations"])
            self.actions.extend(data["actions"])

        # Convert lists to numpy arrays for efficient indexing
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        sample_image = self.observations[idx]
        if self.transform:
            sample_image = self.transform(sample_image)

        if self.action_return:
            sample_action = self.actions[idx]
            if self.action_transform:
                sample_action = self.action_transform(sample_action)
            return sample_image, sample_action
        return sample_image


configs = {
    "VAE": VanillaVAE,
    "hidden_dims": [32, 64, 128, 256],
    "latent_dim": 32,
    "data_dir": "data/dataset/",
    "model_dir": "models/",
    "plot_dir": "plots/",
    "batch_size": 32,
    "num_workers": 6,
    "num_epochs": 50,
    "learning_rate": 3e-4,
    "action_return": False,
    "seed": 42,
}


def train_vae(configs, dataset):
    wandb.init(project="world_models")
    wandb.config.update(configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    vae = configs["VAE"](in_channels=3, **configs).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=configs["learning_rate"])

    dataloader = DataLoader(
        dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
        num_workers=configs["num_workers"],
    )

    for epoch in tqdm(range(configs["num_epochs"])):
        vae.train()
        overall_loss, overall_recons_loss, overall_kld = 0, 0, 0
        for idx, x in enumerate(dataloader):
            if idx % 100 == 0:
                print(f"Batch {idx}/{len(dataloader)}")
            x = x.to(device)
            x_hat, _, mu, log_var = vae(x)
            loss_dict = vae.loss_function(
                x_hat, x, mu, log_var, M_N=configs["batch_size"] / len(dataset)
            )
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()

            overall_loss += loss_dict["loss"].item()
            overall_recons_loss += loss_dict["Reconstruction_Loss"].item()
            overall_kld += loss_dict["KLD"].item()

        wandb.log(
            {
                "Loss": overall_loss / len(dataloader),
                "Recon Loss": overall_recons_loss / len(dataloader),
                "KLD": overall_kld / len(dataloader),
            }
        )
        print(
            f"Epoch {epoch + 1}, Loss: {overall_loss / len(dataloader)}, Recon Loss: {overall_recons_loss / len(dataloader)}, KLD: {overall_kld / len(dataloader)}"
        )

    os.makedirs(configs["model_dir"], exist_ok=True)
    model_name = f"vae_{configs['num_epochs']}.pth"
    torch.save(vae.state_dict(), os.path.join(configs["model_dir"], model_name))


def test_vae(configs, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    vae = configs["VAE"](in_channels=3, **configs).to(device)
    model_name = f"vae_{configs['num_epochs']}.pth"

    vae.load_state_dict(
        torch.load(os.path.join(configs["model_dir"], model_name), map_location=device)
    )
    vae.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
        num_workers=configs["num_workers"],
    )

    for idx, x in enumerate(dataloader):
        x = x.to(device)
        x_hat, _, mu, log_var = vae(x)
        loss_dict = vae.loss_function(
            x_hat, x, mu, log_var, M_N=configs["batch_size"] / len(dataset)
        )
        print(
            f"Loss: {loss_dict['loss'].item()}, Recon Loss: {loss_dict['Reconstruction_Loss'].item()}, KLD: {loss_dict['KLD'].item()}"
        )
        break

    samples = vae.sample(16, device).permute(0, 2, 3, 1).cpu().detach().numpy()
    samples = (samples + 1) / 2

    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(samples[i])
        plt.axis("off")
    plt.show()


def inference(configs, dataset):
    """Show original and reconstructed images together"""
    num_samples = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the VAE and load the trained model weights
    vae = configs["VAE"](
        in_channels=3,
        latent_dim=configs["latent_dim"],
        hidden_dims=configs.get("hidden_dims", None),
    ).to(device)
    model_name = f"vae_{configs['num_epochs']}.pth"
    vae.load_state_dict(
        torch.load(
            os.path.join(configs["model_dir"], model_name),
            map_location=device,
            weights_only=True,
        )
    )
    vae.eval()

    # Sample a batch of data from the dataset
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=num_samples, shuffle=True
    )
    data_iter = iter(data_loader)
    images = next(data_iter)
    images = images.to(device)

    with torch.no_grad():
        recons, _, _, _ = vae(images)

    images = images.cpu()
    recons = recons.cpu()

    # Plot original and reconstructed images
    fig, axs = plt.subplots(2, num_samples, figsize=(20, 4))

    for i in range(num_samples):
        # Show original images
        axs[0, i].imshow((images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1))
        axs[0, i].axis("off")
        # Show reconstructed images
        axs[1, i].imshow((recons[i].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1))
        axs[1, i].axis("off")

    axs[0, 0].set_title("Original Images")
    axs[1, 0].set_title("Reconstructed Images")
    os.makedirs(configs["plot_dir"], exist_ok=True)
    plot_name = "vae_50_reconstruction.png"
    plt.savefig(os.path.join(configs["plot_dir"], plot_name))
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the VAE model")
    parser.add_argument("--test", action="store_true", help="Test the VAE model")
    parser.add_argument(
        "--plot", action="store_true", help="Plot original and reconstructed images"
    )
    args = parser.parse_args()
    seed_everything(configs["seed"])

    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = VAEDataset(
        configs["data_dir"],
        action_return=configs["action_return"],
        transform=transforms,
    )
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    if args.train:
        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        train_vae(configs, train_dataset)
    if args.test:
        test_vae(configs, test_dataset)
    if args.plot:
        inference(configs, test_dataset)
