"""Code is imported from https://github.com/AntixK/PyTorch-VAE/tree/master/models/
Small modifications are made to the original code
"""
import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from types_ import *
from abc import abstractmethod


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
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
    
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_var = nn.Linear(256 * 6 * 6, latent_dim)



        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                                      
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)  # Flatten starting from dimension 1

        print(result.shape)  

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
        print(result.shape)
        result = result.view(-1, 256, 2, 2)
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
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
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

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

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



class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform = None, action_transform = None, action_return = False):
        self.images = np.load(data_dir)['observations']
        self.actions = np.load(data_dir)['actions']
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



configs = {
    'VAE': VanillaVAE,
    'hidden_dims': [32, 64, 128, 256],
    'latent_dim': 32,
    'data_dir': 'dataset/rollout_0.npz',
    'model_dir': 'models/',
    'batch_size': 16,
    'num_workers': 4,
    'num_epochs': 100,
    'learning_rate': 3e-4,
    'action_return': False,
}

def train_vae(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = configs['VAE'](in_channels = 3, **configs)
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=configs['learning_rate'])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = VAEDataset(data_dir = configs['data_dir'], transform = transform, action_return = configs['action_return'])
    dataloader = DataLoader(dataset, batch_size = configs['batch_size'], shuffle = True, num_workers = configs['num_workers'])

    for epoch in range(configs['num_epochs']):
        vae.train()
        overall_loss = 0
        overall_recons_loss = 0
        overall_kld = 0
        for idx, x in enumerate(dataloader):
            x = x.to(device)
            x_hat, _, mu, log_var = vae(x)
            loss_dict = vae.loss_function(x_hat, x, mu, log_var, M_N = configs['batch_size']/len(dataset))
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            overall_loss += loss_dict['loss'].item()
            overall_recons_loss += loss_dict['Reconstruction_Loss'].item()
            overall_kld += loss_dict['KLD'].item()
        print(f"Epoch {epoch+1}, Loss: {overall_loss/len(dataloader)}, Recon Loss: {overall_recons_loss/len(dataloader)}, KLD: {overall_kld/len(dataloader)}")
    
    # Save the model
    os.makedirs(configs['model_dir'], exist_ok = True)
    torch.save(vae.state_dict(), configs['model_dir'] + 'vae.pth')


def test_vae(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = configs['VAE'](in_channels = 3, **configs)
    vae = vae.to(device)
    vae.load_state_dict(torch.load(configs['model_dir'] + 'vae.pth'))
    vae.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = VAEDataset(data_dir = configs['data_dir'], transform = transform, action_return = configs['action_return'])
    dataloader = DataLoader(dataset, batch_size = configs['batch_size'], shuffle = True, num_workers = configs['num_workers'])

    for idx, x in enumerate(dataloader):
        x = x.to(device)
        x_hat, _, mu, log_var = vae(x)
        loss_dict = vae.loss_function(x_hat, x, mu, log_var, M_N = configs['batch_size']/len(dataset))
        print(f"Loss: {loss_dict['loss'].item()}, Recon Loss: {loss_dict['Reconstruction_Loss'].item()}, KLD: {loss_dict['KLD'].item()}")
        break

    # Generate samples
    samples = vae.sample(16, device)
    samples = samples.permute(0, 2, 3, 1).cpu().detach().numpy()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i])
    plt.show()



if __name__ == '__main__':
    # train_vae(configs)
    # test_vae(configs)
    vae = configs['VAE'](in_channels = 3, **configs)
    from torchsummary import summary
    summary(vae, (3, 84, 84))
    print(vae)



