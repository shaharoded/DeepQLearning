# Deep Reinforcement Learning - Assignment 1

Implementation and comparison of Q-Learning algorithms on discrete and continuous control tasks.

## Overview

This project implements three reinforcement learning agents:
1. **Q-Learning** - Tabular method for discrete state spaces (FrozenLake-v1)
2. **Deep Q-Network (DQN)** - Neural network approximation for continuous spaces (CartPole-v1)
3. **Double DQN** - Improved DQN with reduced overestimation bias and optional priority buffer.


## Project Structure
```
Root/
├── src/
│   ├── init.py                     # Package initialization
│   ├── agent.py                    # Agent implementations
│   │ ├── Agent                     # Base class
│   │ ├── QLearningAgent            # Tabular Q-Learning
│   │ ├── DeepQLearningAgent        # Standard DQN
│   │ └── DoubleDeepQLearningAgent  # Double DQN
│   ├── ffnn.py                     # Neural network architectures
│   │ └── QNetwork                  # Feedforward Q-network
│   └── utils.py                    # Replay buffers and utilities
│   ├── ReplayBuffer                # Standard uniform replay
│   └── PrioritizedReplayBuffer     # Priority-based replay
├── train-test-agents.ipynb         # Main notebook for training and evaluation
├── results/                        # Training outputs (plots, summaries, policies)
│   ├── section1/                   # Q-Learning results
│   ├── section2/                   # DQN results
│   └── section3/                   # Double DQN results
├── models/                         # Saved model checkpoints
└── requirements.txt                # Python dependencies
```

## Agents

### Common API
All agents inherit from the `Agent` base class and share a common interface:

**Core Methods:**
- `select_action(state, training=True)` - Choose an action (with exploration if training)
- `update(state, action, reward, next_state, done)` - Learn from a transition
- `train(env, num_episodes, ...)` - Complete training loop
- `evaluate(env, num_episodes=10)` - Evaluate performance
- `save(filepath)` / `load(filepath)` - Persist agent state

**Configuration:**
All agents accept a `config` dictionary with hyperparameters like:
- `learning_rate`: Learning rate for updates
- `gamma`: Discount factor for future rewards
- `epsilon_start/min/decay`: Exploration parameters

>> See train-test-agents.ipynb for usage examples.

### 1. QLearning (Tabular Q-Learning)
Traditional Q-Learning with tabular representation. Suitable for discrete state spaces.

**Key Features:**
- Q-table for state-action values
- ε-greedy exploration
- Temporal Difference learning

### 2. DeepQLearning (DQN)
Deep Q-Network using neural networks to approximate Q-values.

**Key Features:**
- Neural network for function approximation
- Experience replay buffer
- Target network for stable learning
- ε-greedy exploration with decay

### 3. ImprovedDeepQLearning
Enhanced DQN with modern improvements.

**Key Features:**
- Double DQN
- Prioritized experience replay

## Evaluation
Each agent can be evaluated on:
- Training performance (reward over episodes)
- Final policy performance
- Convergence stability

## Results
Results including training curves and performance metrics will be saved in the `results/` directory.

## References
- Sutton & Barto - Reinforcement Learning: An Introduction
- Mnih et al. - Playing Atari with Deep Reinforcement Learning (DQN)
- Van Hasselt et al. - Deep Reinforcement Learning with Double Q-learning
- Wang et al. - Dueling Network Architectures for Deep Reinforcement Learning
