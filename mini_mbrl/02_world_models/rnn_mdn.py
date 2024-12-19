"""RNN+MDN model for predicting next latent state given current latent state, action, and hidden state.
MDN is used to model the distribution of the next latent state given the current latent state and action(Bishop, 1994)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MDNRNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians, num_layers=1):
        super(MDNRNN, self).__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=latent_dim + action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.pi = nn.Linear(hidden_dim, num_gaussians)
        self.mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self, latent_seq, action_seq, hidden=None):
        rnn_input = torch.cat([latent_seq, action_seq], dim=2)
        rnn_output, hidden = self.rnn(rnn_input, hidden)

        pi = F.softmax(self.pi(rnn_output), dim=2)
        mu = self.mu(rnn_output)
        sigma = torch.exp(self.sigma(rnn_output))

        return pi, mu, sigma, hidden

    def get_mixture_loss(self, pi, mu, sigma, target):
        """
        Compute the negative log likelihood loss for a mixture of Gaussians.
        Args:
            pi: (batch_size, seq_len, num_gaussians)
            mu: (batch_size, seq_len, num_gaussians * latent_dim)
            sigma: (batch_size, seq_len, num_gaussians * latent_dim)
            target: (batch_size, seq_len, latent_dim)
        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, _ = target.shape

        # Reshape parameters
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        sigma = sigma.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        # Keep pi as (batch_size, seq_len, num_gaussians)
        # No need to unsqueeze pi here

        # Expand target
        target = target.unsqueeze(2).expand(-1, -1, self.num_gaussians, -1)

        # Compute log probability for each Gaussian
        z_score = (target - mu) / (sigma + 1e-8)
        log_p = -0.5 * (
            torch.log(2 * torch.tensor(np.pi, device=target.device)) + 2 * torch.log(sigma + 1e-8) + z_score.pow(2)
        )

        # Sum over latent dimensions
        log_p = torch.sum(log_p, dim=-1)  # (batch_size, seq_len, num_gaussians)

        # Add log mixing coefficients
        log_pi = torch.log(pi + 1e-8)  # (batch_size, seq_len, num_gaussians)
        log_prob_mix = log_pi + log_p

        # Log-sum-exp trick
        max_log_prob = torch.max(log_prob_mix, dim=-1)[0]  # remove keepdim=True
        log_sum = torch.log(torch.sum(torch.exp(log_prob_mix - max_log_prob.unsqueeze(-1)), dim=-1))
        log_prob = max_log_prob + log_sum

        # Average over batch and sequence dimensions
        loss = -torch.mean(log_prob)

        # Add debugging information
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"pi shape: {pi.shape}")
            print(f"mu shape: {mu.shape}")
            print(f"sigma shape: {sigma.shape}")
            print(f"target shape: {target.shape}")
            print(f"log_prob_mix shape: {log_prob_mix.shape}")
            print(f"max_log_prob shape: {max_log_prob.shape}")
            print(f"log_sum shape: {log_sum.shape}")

        return loss


class MDNRNNDataset(Dataset):
    def __init__(self, latent_sequences, action_sequences, sequence_length):
        """
        Args:
            latent_sequences: List of latent state sequences
            action_sequences: List of action sequences
            sequence_length: Length of sequences to generate
        """
        self.latent_sequences = latent_sequences
        self.action_sequences = action_sequences
        self.sequence_length = sequence_length

        # Create sliding windows of sequences
        self.samples = []
        for latent_seq, action_seq in zip(latent_sequences, action_sequences):
            if len(latent_seq) < sequence_length + 1:  # +1 for next latent
                continue
            for i in range(len(latent_seq) - sequence_length):
                self.samples.append(
                    {
                        "latent": latent_seq[i : i + sequence_length],
                        "action": action_seq[i : i + sequence_length],
                        "next_latent": latent_seq[i + 1 : i + sequence_length + 1],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "latent": torch.FloatTensor(sample["latent"]),
            "action": torch.FloatTensor(sample["action"]),
            "next_latent": torch.FloatTensor(sample["next_latent"]),
        }


def debug_shapes(pi, mu, sigma, target):
    """Helper function to debug tensor shapes"""
    print(f"pi shape: {pi.shape}")
    print(f"mu shape: {mu.shape}")
    print(f"sigma shape: {sigma.shape}")
    print(f"target shape: {target.shape}")
    print(f"pi range: [{pi.min()}, {pi.max()}]")
    print(f"mu range: [{mu.min()}, {mu.max()}]")
    print(f"sigma range: [{sigma.min()}, {sigma.max()}]")
