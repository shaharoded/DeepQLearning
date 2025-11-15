"""
Feed-forward neural network architectures for Q-value approximation.
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    
    Simple feedforward network with configurable architecture.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions (e.g., [64, 64, 64] for 3 layers)
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Store architecture info
        self.hidden_dims = hidden_dims
        self.num_hidden_layers = len(hidden_dims)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Q-values for all actions (batch_size, action_dim) or (action_dim,)
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.
    
    Separates the representation of state value V(s) and advantage A(s,a):
    Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize dueling Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions
        """
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature layers
        feature_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Store architecture info
        self.hidden_dims = hidden_dims
        self.num_hidden_layers = len(hidden_dims)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling network.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for all actions
        """
        features = self.feature_layer(state)
        
        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values