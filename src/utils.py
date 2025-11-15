"""
Utility classes and functions for DQN agents.
"""

import numpy as np
import torch
import random
from collections import deque, namedtuple
from typing import Tuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience replay buffer for DQN agents.
    Stores transitions and samples random minibatches for training.
    
    This buffer is updated as new experiences are collected.

    When? After each step, if the buffer has enough samples (≥ batch_size), 
    we sample a random batch and perform one gradient update

    NOTE: This is a simple uniform replay buffer. A possible improvement will implement prioritized experience replay.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)