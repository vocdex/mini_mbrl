# This file contains the dynamics model used in the CEM MPC algorithm
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Initialize weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        # Ensure consistent dimensions (batch size of 1)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() == 3:
            action = action.squeeze(1)

        x = torch.cat([state, action], dim=-1)
        delta = self.network(x)
        next_state = state + delta
        return next_state


class MLPDynamicsModel:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.model = MLP(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def train(self, states, actions, next_states):
        """One step of forward-backward pass on the dynamics model."""
        self.optimizer.zero_grad()
        predicted_next_states = self.model(states, actions)
        loss = self.loss_fn(predicted_next_states, next_states)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_1_step(self, state, action):
        """Use the model to predict one step into the future."""
        # Ensure inputs are tensors
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)

        # Handle single state/action inputs
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        return self.model(state, action)  # Torch Tensor: [batch_size, state_dim]

    def predict_n_steps(self, states, action_sequences, n):
        """Use the model to predict n steps into the future.

        Args:
            states (torch.Tensor): Initial states tensor
            actions (torch.Tensor): Sequence of actions to take
            n (int): Number of steps to predict forward

        Returns:
            torch.Tensor: Predicted states for each step
        """
        # Ensure inputs are tensors
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        if not isinstance(action_sequences, torch.Tensor):
            action_sequences = torch.FloatTensor(action_sequences)

        # Handle single state/action inputs
        if states.dim() == 1:
            states = states.unsqueeze(0)  # Add batch dimension
        if action_sequences.dim() == 1:
            action_sequences = action_sequences.unsqueeze(0)  # Add batch dimension

        # Ensure we have enough actions for n steps
        assert action_sequences.shape[1] >= n, f"Need at least {n} actions, but got {action_sequences.shape[1]}"

        # Initialize list to store predicted states
        predicted_states = [states]

        # Predict n steps forward
        current_state = states
        with torch.no_grad():  # No need for gradients during prediction
            for i in range(n):
                current_action = action_sequences[:, i : i + 1]  # Get action for current step
                next_state = self.predict_1_step(current_state, current_action)
                predicted_states.append(next_state)
                current_state = next_state

        # Stack all predicted states
        predicted_states = torch.stack(predicted_states, dim=1)  # Shape: [batch_size, n+1, state_dim]

        return predicted_states

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
