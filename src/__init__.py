"""
Deep Reinforcement Learning Assignment - Source Module
"""

from .agent import (
    Agent,
    QLearningAgent,
    DeepQLearningAgent,
    ImprovedDeepQLearningAgent
)

__all__ = [
    'Agent',
    'QLearningAgent',
    'DeepQLearningAgent',
    'ImprovedDeepQLearningAgent'
]

__version__ = '1.0.0'
