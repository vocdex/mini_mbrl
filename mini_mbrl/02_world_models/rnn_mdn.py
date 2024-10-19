"""RNN+MDN model for predicting next latent state given current latent state, action, and hidden state.
MDN is used to model the distribution of the next latent state given the current latent state and action(Bishop, 1994)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class RNN_MDN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, num_gaussians):
        super(RNN_MDN, self).__init__()
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        self.latent_dim = latent_dim

        # RNN layer (LSTM or GRU can be used)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Output layer for MDN (mu, sigma, pi for each Gaussian)
        self.fc = nn.Linear(hidden_size, num_gaussians * (2 * latent_dim + 1))

    def forward(self, x, hidden=None):
        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)
        # Output layer for mixture density network
        params = self.fc(out)
        return params, hidden

def mdn_split_params(params, latent_dim, num_gaussians):
    """
    Split the MDN output into means (mu), standard deviations (sigma), and mixing coefficients (pi).
    """
    pi = params[..., :num_gaussians]
    mu = params[..., num_gaussians:num_gaussians + num_gaussians * latent_dim].view(-1, num_gaussians, latent_dim)
    sigma = params[..., num_gaussians + num_gaussians * latent_dim:].view(-1, num_gaussians, latent_dim)
    return pi, mu, sigma


def mdn_loss(pi, mu, sigma, y):
    """
    Computes the MDN loss (Negative Log-Likelihood).
    - pi: Mixing coefficients (B x num_gaussians)
    - mu: Means of the Gaussians (B x num_gaussians x latent_dim)
    - sigma: Standard deviations of the Gaussians (B x num_gaussians x latent_dim)
    - y: Ground truth latent state (B x latent_dim)
    """
    # Convert sigma to a positive value
    sigma = torch.exp(sigma)

    # Compute the exponent of the Gaussian
    m = torch.distributions.Normal(mu, sigma)
    log_prob = m.log_prob(y.unsqueeze(1))  # Broadcast to match Gaussian shape
    log_prob = log_prob.sum(dim=2)  # Sum over latent dimensions

    # Compute log likelihood of the mixture model
    log_pi = F.log_softmax(pi, dim=1)  # Softmax on pi to get valid probabilities
    log_sum_exp = torch.logsumexp(log_pi + log_prob, dim=1)

    # Return the negative log-likelihood
    return -log_sum_exp.mean()


class RolloutDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.latent_states = data['latent_states']  # Shape (num_episodes, T, latent_dim)
        self.actions = data['actions']  # Shape (num_episodes, T, action_dim)

    def __len__(self):
        return len(self.latent_states)

    def __getitem__(self, idx):
        return torch.tensor(self.latent_states[idx], dtype=torch.float32), \
               torch.tensor(self.actions[idx], dtype=torch.float32)

# Usage example
dataset = RolloutDataset('rollouts.npz')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

