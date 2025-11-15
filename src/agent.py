"""
Agent implementations for Deep Reinforcement Learning Assignment.

This module contains the base Agent class and three implementations:
- QLearningAgent: Traditional tabular Q-Learning
- DeepQLearningAgent: Deep Q-Network (DQN)
- ImprovedDeepQLearningAgent: Enhanced DQN with modern improvements
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional, Dict, Any
import gymnasium as gym

# Local imports
from src.utils import ReplayBuffer
from src.ffnn import QNetwork, DuelingQNetwork


class Agent:
    """
    Base Agent class defining the common interface for all RL agents.
    
    All agents must implement the core methods for interaction, learning, and evaluation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            config: Optional configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Common hyperparameters
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # Exploration parameters
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)  # Back to exponential
        
        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            training: Whether in training mode (enables exploration)
            
        Returns:
            Selected action index
        """
        raise NotImplementedError("Subclasses must implement select_action()")
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """
        Update the agent's knowledge based on a transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            Loss value if applicable, None otherwise
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def train_agent(self, env: gym.Env, num_episodes: int = None, max_steps: int = 1000,
                    eval_frequency: int = 100, verbose: bool = True,
                    save_q_table_at: Optional[List[int]] = None,
                    target_reward: float = 475.0, target_window: int = 100,
                    max_episodes: int = 2000) -> Dict[str, Any]:
        """
        Train the agent in the given environment with early stopping.
        
        Args:
            env: Gymnasium environment
            num_episodes: DEPRECATED - use max_episodes instead
            max_steps: Maximum steps per episode
            eval_frequency: Evaluate every N episodes
            verbose: Print training progress
            save_q_table_at: List of episode numbers to save Q-table (for QLearningAgent)
            target_reward: Target average reward for early stopping (default: 475.0)
            target_window: Window size for computing average (default: 100)
            max_episodes: Maximum episodes before stopping (default: 2000)
            
        Returns:
            Dictionary containing training metrics
        """
        # Backward compatibility
        if num_episodes is not None:
            max_episodes = num_episodes
            
        # Store Q-tables
        q_tables_log: Dict[int, np.ndarray] = {}
        converged = False
        
        for episode in range(max_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            step_count = 0
            
            for step in range(max_steps):
                # Select and execute action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Update agent
                loss = self.update(state, action, reward, next_state, done)
                if loss is not None:
                    episode_loss += loss
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                if done:
                    break
            
            # Store statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)
            if episode_loss > 0:
                self.losses.append(episode_loss / step_count)
            
            # Decay exploration
            self._decay_epsilon()

            # Save Q-table if requested
            if save_q_table_at and (episode + 1) in save_q_table_at:
                # Check if this is the QLearningAgent
                if hasattr(self, 'q_table'):
                    q_tables_log[episode + 1] = np.copy(self.q_table)
            
            # Check early stopping criterion
            if len(self.episode_rewards) >= target_window:
                avg_reward = np.mean(self.episode_rewards[-target_window:])
                if avg_reward >= target_reward:
                    converged = True
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"EARLY STOPPING: Target achieved!")
                        print(f"Episode {episode + 1}: {target_window}-episode average = {avg_reward:.2f} ≥ {target_reward}")
                        print(f"{'='*60}\n")
                    break
            
            # Logging
            if verbose and (episode + 1) % eval_frequency == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_frequency:])
                avg_length = np.mean(self.episode_lengths[-eval_frequency:])
                print(f"Episode {episode + 1}/{max_episodes} | "
                      f"Avg Reward (=Length): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'losses': self.losses,
            'q_tables': q_tables_log,
            'converged': converged,
            'episodes_trained': len(self.episode_rewards)
        }
    
    def test_agent(self, env: gym.Env, num_episodes: int = 10, 
                   render: bool = False, max_steps: int = 1000) -> Tuple[float, float]:
        """
        Test the agent on new episodes with the trained model.
        
        This is a general implementation that works for all agents.
        Can be overridden by subclasses if needed.
        
        Args:
            env: Gymnasium environment
            num_episodes: Number of test episodes
            render: Whether to render the environment
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        rewards = []
        lengths = []
        
        print(f"\nTesting agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            steps = 0
            
            for step in range(max_steps):
                if render:
                    env.render()
                
                # Use greedy policy (no exploration)
                action = self.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            rewards.append(episode_reward)
            lengths.append(steps)
            
            if render:
                print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_length = np.mean(lengths)
        
        print(f"\nTest Results:")
        print(f"Mean Reward (= Length): {mean_reward:.2f} ± {std_reward:.2f}")
        
        return mean_reward, std_reward
    
    def save(self, filepath: str):
        """Save agent parameters to file."""
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, filepath: str):
        """Load agent parameters from file."""
        raise NotImplementedError("Subclasses must implement load()")
    
    def _decay_epsilon(self):
        """Decay exploration rate using exponential decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningAgent(Agent):
    """
    Traditional Q-Learning agent with tabular representation.
    
    Suitable for discrete state and action spaces.
    Uses temporal difference learning to update Q-values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_dim: Dimension of discretized state space (or number of states)
            action_dim: Number of possible actions
            config: Configuration dictionary
        """
        super().__init__(state_dim, action_dim, config)
        
        # Q-table initialization
        self.q_table = np.zeros((state_dim, action_dim))
        
        # Q-Learning specific parameters
        self.alpha = self.config.get('alpha', 0.1)  # Learning rate for Q-table
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (should be discrete index or discretized)
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        # Convert state to discrete index if needed
        state_idx = self._get_state_index(state)
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """
        Update Q-table using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            TD error (as pseudo-loss)
        """
        state_idx = self._get_state_index(state)
        next_state_idx = self._get_state_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_idx, action]
        
        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_idx])
        
        # TD error
        td_error = target_q - current_q
        
        # Q-table update
        self.q_table[state_idx, action] += self.alpha * td_error
        
        return abs(td_error)
    
    def _get_state_index(self, state: np.ndarray) -> int:
        """
        Convert state to discrete index.
        
        For continuous states, this should discretize them.
        For already discrete states, return as-is.
        
        Args:
            state: State observation
            
        Returns:
            Discrete state index
        """
        # If state is already a single integer
        if isinstance(state, (int, np.integer)):
            return int(state)
        
        # If state is array with single element
        if state.size == 1:
            return int(state.item())
        
        # For continuous states, implement discretization
        # This is a placeholder - should be customized based on environment
        # TODO: Implement proper state discretization for continuous spaces
        return 0
    
    def save(self, filepath: str):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from {filepath}")


class DeepQLearningAgent(Agent):
    """
    Deep Q-Network (DQN) agent.
    
    Uses neural network for Q-value approximation with experience replay (from a buffer)
    and target network for stable learning.
    
    This implementation follows the DQN algorithm:
    1. Initialize replay buffer and Q-networks
    2. For each episode:
        - Select action using epsilon-greedy
        - Execute action and observe reward and next state
        - Store transition in replay buffer
        - Sample minibatch from replay buffer (if the buffer has enough samples)
        - Compute target Q-values using target network
        - Update Q-network to minimize loss (using ADAM optimizer and SmoothL1Loss() huber loss criterion)
        - Periodically update target network with Q-network weights, to stabilize learning (by stabilizing target values).
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            config: Configuration dictionary with keys:
                - batch_size: Minibatch size for training (default: 64)
                - buffer_capacity: Replay buffer size (default: 10000)
                - target_update_freq: Steps between target network updates (default: 100)
                - hidden_dims: List of hidden layer sizes (default: [128, 128])
                - learning_rate: Learning rate for optimizer (default: 0.001)
                - gamma: Discount factor (default: 0.99)
                - epsilon_start: Initial exploration rate (default: 1.0)
                - epsilon_min: Minimum exploration rate (default: 0.01)
        """
        super().__init__(state_dim, action_dim, config)
        
        # DQN specific parameters
        self.batch_size = self.config.get('batch_size', 64)
        self.buffer_capacity = self.config.get('buffer_capacity', 10000)
        self.target_update_freq = self.config.get('target_update_freq', 100)
        self.hidden_dims = self.config.get('hidden_dims', [128, 128])
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim, self.hidden_dims) # Updates every training step
        self.target_network = QNetwork(state_dim, action_dim, self.hidden_dims) # Target network updated less frequently
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Ensure networks are in correct mode
        self.q_network.train()  # Training mode
        self.target_network.eval()  # Eval mode (no gradients needed)
        
        print(f"Initialized DQN with {self.q_network.num_hidden_layers} hidden layers: {self.hidden_dims}")
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        
        # Training step counter
        self.training_steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with neural network.
        
        Args:
            state: Current state
            training: Whether in training mode (enables epsilon-greedy exploration)
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Greedy action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item() # Select action with highest Q-value
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """
        Store transition and perform learning update if buffer is ready.
        
        This implements the DQN learning step:
        1. Store experience in replay buffer
        2. Sample random minibatch
        3. Compute target Q-values: r + γ max_a' Q_target(s', a')
        4. Update Q-network to minimize loss
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            Loss value if update was performed, None otherwise
        """
        # Store transition in replay buffer (experience_replay)
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Only update if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample minibatch (sample_batch)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Ensure Q-network is in training mode
        self.q_network.train()
        
        # Compute current Q-values Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values: y = r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Detach target to prevent gradient flow
        target_q_values = target_q_values.detach()
        
        # Compute loss: L = (Q(s,a) - y)^2
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save network parameters to file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load network parameters from file."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")


class ImprovedDeepQLearningAgent(DeepQLearningAgent):
    """
    Improved DQN agent with modern enhancements.
    
    Implements:
    - Double DQN: Reduces overestimation bias
    - Dueling architecture: Separates state value and advantage
    - Prioritized experience replay: Samples important transitions more frequently
    - Additional improvements can be added (e.g., noisy networks, distributional RL)
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize improved DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            config: Configuration dictionary
        """
        # Initialize parent (will be customized below)
        super().__init__(state_dim, action_dim, config)
        
        # Improvement flags
        self.use_double_dqn = self.config.get('use_double_dqn', True)
        self.use_dueling = self.config.get('use_dueling', True)
        self.use_prioritized_replay = self.config.get('use_prioritized_replay', False)
        
        # Replace networks with dueling architecture if enabled
        if self.use_dueling:
            self.q_network = DuelingQNetwork(state_dim, action_dim, self.hidden_dims)
            self.target_network = DuelingQNetwork(state_dim, action_dim, self.hidden_dims)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Reinitialize optimizer with new network
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # TODO: Implement prioritized experience replay
        if self.use_prioritized_replay:
            print("Note: Prioritized replay not yet implemented, using standard replay.")
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Optional[float]:
        """
        Update with Double DQN and other improvements.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            Loss value if update was performed, None otherwise
        """
        # Store transition
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Use online network for action selection, target network for evaluation
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save network parameters to file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load network parameters from file."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage and testing
    print("Agent classes loaded successfully!")
    print("\nAvailable agents:")
    print("1. QLearningAgent - Traditional tabular Q-Learning")
    print("2. DeepQLearningAgent - Deep Q-Network (DQN)")
    print("3. ImprovedDeepQLearningAgent - Enhanced DQN with Double DQN and Dueling architecture")
    
    # Quick test
    print("\nQuick initialization test:")
    try:
        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        print(f"\nEnvironment: CartPole-v1")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        
        agent1 = QLearningAgent(state_dim=100, action_dim=action_dim)
        print("\n✓ QLearningAgent initialized")
        
        # Test with 3 hidden layers
        config_3_layers = {'hidden_dims': [64, 64, 64]}
        agent2_3layers = DeepQLearningAgent(state_dim=state_dim, action_dim=action_dim, 
                                           config=config_3_layers)
        print("✓ DeepQLearningAgent initialized (3 hidden layers)")
        
        # Test with 5 hidden layers
        config_5_layers = {'hidden_dims': [64, 64, 64, 64, 64]}
        agent2_5layers = DeepQLearningAgent(state_dim=state_dim, action_dim=action_dim,
                                           config=config_5_layers)
        print("✓ DeepQLearningAgent initialized (5 hidden layers)")
        
        agent3 = ImprovedDeepQLearningAgent(state_dim=state_dim, action_dim=action_dim)
        print("✓ ImprovedDeepQLearningAgent initialized")
        
        print("\nAll agents initialized successfully!")
        print("\nTo train an agent, use:")
        print("  results = agent.train_agent(env, num_episodes=500)")
        print("\nTo test an agent, use:")
        print("  mean_reward, std_reward = agent.test_agent(env, num_episodes=10, render=True)")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()
