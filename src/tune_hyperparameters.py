"""
Hyperparameter Tuning for Q-Learning Agent on FrozenLake-v1.

This script performs a systematic search over hyperparameters to find
the best configuration for solving the FrozenLake environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from typing import Dict, List, Tuple, Any
import pandas as pd
from itertools import product
import os
from tqdm import tqdm
import json

from src.agent import QLearningAgent


def evaluate_agent_performance(agent: QLearningAgent, env: gym.Env, 
                               num_episodes: int = 100) -> Tuple[float, float]:
    """
    Evaluate the trained agent's performance.
    
    Args:
        agent: Trained QLearningAgent
        env: Gymnasium environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Tuple of (success_rate, avg_reward)
    """
    successes = 0
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, training=False)  # Greedy policy
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        if reward > 0:  # Goal reached in FrozenLake
            successes += 1
        total_rewards.append(episode_reward)
    
    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    
    return success_rate, avg_reward


def train_and_evaluate(config: Dict[str, Any], num_episodes: int = 10000,
                       max_steps: int = 100, eval_episodes: int = 100,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    Train a Q-Learning agent with given hyperparameters and evaluate performance.
    
    Args:
        config: Hyperparameter configuration dictionary
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_episodes: Number of episodes for evaluation
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training results and performance metrics
    """
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # Initialize agent
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = QLearningAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    
    # Train
    metrics = agent.train(env, num_episodes=num_episodes, max_steps=max_steps, 
                         verbose=verbose, eval_frequency=1000)
    
    # Evaluate
    success_rate, avg_reward = evaluate_agent_performance(agent, env, eval_episodes)
    
    env.close()
    
    return {
        'config': config,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'final_epsilon': agent.epsilon,
        'training_rewards': metrics['rewards'],
        'training_lengths': metrics['lengths']
    }


def grid_search_hyperparameters(param_grid: Dict[str, List], 
                                num_episodes: int = 10000,
                                max_steps: int = 100,
                                eval_episodes: int = 100,
                                n_trials: int = 3) -> pd.DataFrame:
    """
    Perform grid search over hyperparameter space.
    
    Args:
        param_grid: Dictionary of parameter names to list of values to try
        num_episodes: Number of training episodes per configuration
        max_steps: Maximum steps per episode
        eval_episodes: Number of episodes for evaluation
        n_trials: Number of trials per configuration (for statistical reliability)
        
    Returns:
        DataFrame with results for all configurations
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\n{'='*70}")
    print(f"GRID SEARCH: Testing {len(combinations)} configurations × {n_trials} trials")
    print(f"{'='*70}\n")
    
    results = []
    
    # Progress bar for all configurations
    for idx, combination in enumerate(tqdm(combinations, desc="Grid Search Progress")):
        config = dict(zip(param_names, combination))
        
        # Run multiple trials for this configuration
        trial_results = []
        for trial in range(n_trials):
            result = train_and_evaluate(
                config=config,
                num_episodes=num_episodes,
                max_steps=max_steps,
                eval_episodes=eval_episodes,
                verbose=False
            )
            trial_results.append({
                'success_rate': result['success_rate'],
                'avg_reward': result['avg_reward']
            })
        
        # Aggregate trial results
        avg_success_rate = np.mean([r['success_rate'] for r in trial_results])
        std_success_rate = np.std([r['success_rate'] for r in trial_results])
        avg_reward = np.mean([r['avg_reward'] for r in trial_results])
        std_reward = np.std([r['avg_reward'] for r in trial_results])
        
        results.append({
            **config,
            'success_rate_mean': avg_success_rate,
            'success_rate_std': std_success_rate,
            'avg_reward_mean': avg_reward,
            'avg_reward_std': std_reward,
            'n_trials': n_trials
        })
        
        # Print progress for this configuration
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"\n[Config {idx+1}/{len(combinations)}] "
                  f"Success Rate: {avg_success_rate:.3f} ± {std_success_rate:.3f}")
    
    df = pd.DataFrame(results)
    return df


def plot_hyperparameter_heatmaps(df: pd.DataFrame, save_dir: str = 'results/tuning'):
    """
    Create heatmap visualizations for hyperparameter effects.
    
    Args:
        df: DataFrame with tuning results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("viridis")
    
    # 1. Alpha vs Gamma Heatmap
    if 'alpha' in df.columns and 'gamma' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Success Rate Heatmap
        pivot_success = df.pivot_table(
            values='success_rate_mean', 
            index='alpha', 
            columns='gamma'
        )
        sns.heatmap(pivot_success, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[0], vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        axes[0].set_title('Success Rate: Alpha vs Gamma', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Gamma (Discount Factor)', fontsize=12)
        axes[0].set_ylabel('Alpha (Learning Rate)', fontsize=12)
        
        # Average Reward Heatmap
        pivot_reward = df.pivot_table(
            values='avg_reward_mean', 
            index='alpha', 
            columns='gamma'
        )
        sns.heatmap(pivot_reward, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[1], cbar_kws={'label': 'Avg Reward'})
        axes[1].set_title('Average Reward: Alpha vs Gamma', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Gamma (Discount Factor)', fontsize=12)
        axes[1].set_ylabel('Alpha (Learning Rate)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/heatmap_alpha_gamma.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: {save_dir}/heatmap_alpha_gamma.png")
    
    # 2. Epsilon Decay Heatmap (if epsilon_decay varies)
    if 'epsilon_decay' in df.columns and df['epsilon_decay'].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Success Rate vs Epsilon Decay and Alpha
        if 'alpha' in df.columns:
            pivot_success = df.pivot_table(
                values='success_rate_mean', 
                index='epsilon_decay', 
                columns='alpha'
            )
            sns.heatmap(pivot_success, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=axes[0], vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
            axes[0].set_title('Success Rate: Epsilon Decay vs Alpha', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Alpha (Learning Rate)', fontsize=12)
            axes[0].set_ylabel('Epsilon Decay', fontsize=12)
        
        # Success Rate vs Epsilon Decay and Gamma
        if 'gamma' in df.columns:
            pivot_reward = df.pivot_table(
                values='success_rate_mean', 
                index='epsilon_decay', 
                columns='gamma'
            )
            sns.heatmap(pivot_reward, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
            axes[1].set_title('Success Rate: Epsilon Decay vs Gamma', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Gamma (Discount Factor)', fontsize=12)
            axes[1].set_ylabel('Epsilon Decay', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/heatmap_epsilon_decay.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: {save_dir}/heatmap_epsilon_decay.png")


def plot_parameter_effects(df: pd.DataFrame, save_dir: str = 'results/tuning'):
    """
    Create box plots and violin plots showing parameter effects.
    
    Args:
        df: DataFrame with tuning results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    params = ['alpha', 'gamma', 'epsilon_decay']
    available_params = [p for p in params if p in df.columns and df[p].nunique() > 1]
    
    if not available_params:
        print("Not enough parameter variation for effect plots")
        return
    
    n_params = len(available_params)
    fig, axes = plt.subplots(n_params, 2, figsize=(14, 5*n_params))
    
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    for idx, param in enumerate(available_params):
        # Success Rate
        sns.boxplot(data=df, x=param, y='success_rate_mean', ax=axes[idx, 0])
        axes[idx, 0].set_title(f'Success Rate vs {param.capitalize()}', 
                               fontsize=12, fontweight='bold')
        axes[idx, 0].set_ylabel('Success Rate', fontsize=11)
        axes[idx, 0].set_xlabel(param.capitalize(), fontsize=11)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Average Reward
        sns.boxplot(data=df, x=param, y='avg_reward_mean', ax=axes[idx, 1])
        axes[idx, 1].set_title(f'Average Reward vs {param.capitalize()}', 
                               fontsize=12, fontweight='bold')
        axes[idx, 1].set_ylabel('Average Reward', fontsize=11)
        axes[idx, 1].set_xlabel(param.capitalize(), fontsize=11)
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {save_dir}/parameter_effects.png")


def create_top_configurations_table(df: pd.DataFrame, top_n: int = 5, 
                                   save_dir: str = 'results/tuning'):
    """
    Create a formatted table of top N configurations.
    
    Args:
        df: DataFrame with tuning results
        top_n: Number of top configurations to show
        save_dir: Directory to save table
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort by success rate
    df_sorted = df.sort_values('success_rate_mean', ascending=False).head(top_n)
    
    # Create a clean table
    table_data = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        table_data.append({
            'Rank': rank,
            'Alpha (α)': f"{row['alpha']:.3f}",
            'Gamma (γ)': f"{row['gamma']:.2f}",
            'Epsilon Decay': f"{row['epsilon_decay']:.4f}",
            'Success Rate': f"{row['success_rate_mean']:.3f} ± {row['success_rate_std']:.3f}",
            'Avg Reward': f"{row['avg_reward_mean']:.3f} ± {row['avg_reward_std']:.3f}"
        })
    
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    table_df.to_csv(f'{save_dir}/top_configurations_table.csv', index=False)
    print(f"✓ Saved table to: {save_dir}/top_configurations_table.csv")
    
    # Print formatted table to console
    print(f"\n{'='*90}")
    print(f"TOP {top_n} CONFIGURATIONS BY SUCCESS RATE")
    print(f"{'='*90}")
    print(table_df.to_string(index=False))
    print(f"{'='*90}\n")
    
    return table_df


def plot_top_configurations(df: pd.DataFrame, top_n: int = 5, 
                           save_dir: str = 'results/section1/tuning'):
    """
    Plot comparison of top N configurations.
    
    Args:
        df: DataFrame with tuning results
        top_n: Number of top configurations to show
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort by success rate
    df_sorted = df.sort_values('success_rate_mean', ascending=False).head(top_n)
    
    # Create labels for configurations
    labels = []
    for _, row in df_sorted.iterrows():
        label = f"α={row['alpha']:.3f}, γ={row['gamma']:.2f}"
        if 'epsilon_decay' in df.columns:
            label += f", ε_decay={row['epsilon_decay']:.4f}"
        labels.append(label)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Success Rate
    axes[0].barh(labels, df_sorted['success_rate_mean'], 
                 xerr=df_sorted['success_rate_std'], 
                 color=sns.color_palette("viridis", top_n))
    axes[0].set_xlabel('Success Rate', fontsize=12)
    axes[0].set_title(f'Top {top_n} Configurations by Success Rate', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_xlim(0, 1)
    
    # Average Reward
    axes[1].barh(labels, df_sorted['avg_reward_mean'], 
                 xerr=df_sorted['avg_reward_std'],
                 color=sns.color_palette("viridis", top_n))
    axes[1].set_xlabel('Average Reward', fontsize=12)
    axes[1].set_title(f'Top {top_n} Configurations by Average Reward', 
                     fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/top_configurations.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {save_dir}/top_configurations.png")


def plot_learning_curves_comparison(configs: List[Dict], num_episodes: int = 10000,
                                   max_steps: int = 100, save_dir: str = 'results/tuning'):
    """
    Plot learning curves for multiple configurations.
    
    Args:
        configs: List of configuration dictionaries to compare
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 100
    
    colors = sns.color_palette("husl", len(configs))
    
    for idx, config in enumerate(configs):
        result = train_and_evaluate(config, num_episodes, max_steps, eval_episodes=100, verbose=False)
        rewards = result['training_rewards']
        lengths = result['training_lengths']
        
        # Create label
        label = f"α={config['alpha']:.3f}, γ={config['gamma']:.2f}"
        if 'epsilon_decay' in config:
            label += f", ε_d={config['epsilon_decay']:.4f}"
        label += f" (SR={result['success_rate']:.3f})"
        
        # Plot rewards
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards)), moving_avg, 
                        label=label, linewidth=2, color=colors[idx])
        
        # Plot episode lengths
        if len(lengths) >= window:
            moving_avg_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(lengths)), moving_avg_lengths, 
                        label=label, linewidth=2, color=colors[idx])
    
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Reward (Moving Average)', fontsize=12)
    axes[0].set_title('Training Rewards Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Episode Length (Moving Average)', fontsize=12)
    axes[1].set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {save_dir}/learning_curves_comparison.png")


def main():
    """Main hyperparameter tuning workflow."""
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING FOR Q-LEARNING ON FROZENLAKE-V1")
    print("="*70 + "\n")
    
    # Define hyperparameter search space
    param_grid = {
        'alpha': [0.05, 0.1, 0.2, 0.3],              # Learning rate
        'gamma': [0.9, 0.95, 0.99],                   # Discount factor
        'epsilon_start': [1.0],                       # Always start with full exploration
        'epsilon_min': [0.01],                        # Minimum exploration
        'epsilon_decay': [0.999, 0.9995, 0.9998]     # Exploration decay rate
    }
    
    print("Search Space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()
    
    # Perform grid search
    results_df = grid_search_hyperparameters(
        param_grid=param_grid,
        num_episodes=10000,
        max_steps=100,
        eval_episodes=100,
        n_trials=3  # Multiple trials for reliability
    )
    
    # Save results
    save_dir = 'results/tuning'
    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(f'{save_dir}/tuning_results.csv', index=False)
    print(f"\n✓ Results saved to: {save_dir}/tuning_results.csv")
    
    # Find and display best configuration
    best_config = results_df.loc[results_df['success_rate_mean'].idxmax()]
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION FOUND")
    print("="*70)
    print(f"\nHyperparameters:")
    print(f"  Alpha (Learning Rate):     {best_config['alpha']:.4f}")
    print(f"  Gamma (Discount Factor):   {best_config['gamma']:.4f}")
    print(f"  Epsilon Start:             {best_config['epsilon_start']:.4f}")
    print(f"  Epsilon Min:               {best_config['epsilon_min']:.4f}")
    print(f"  Epsilon Decay:             {best_config['epsilon_decay']:.4f}")
    print(f"\nPerformance:")
    print(f"  Success Rate:  {best_config['success_rate_mean']:.3f} ± {best_config['success_rate_std']:.3f}")
    print(f"  Average Reward: {best_config['avg_reward_mean']:.3f} ± {best_config['avg_reward_std']:.3f}")
    print("="*70 + "\n")
    
    # Save best config as JSON
    best_config_dict = {
        'alpha': float(best_config['alpha']),
        'gamma': float(best_config['gamma']),
        'epsilon_start': float(best_config['epsilon_start']),
        'epsilon_min': float(best_config['epsilon_min']),
        'epsilon_decay': float(best_config['epsilon_decay'])
    }
    
    with open(f'{save_dir}/best_config.json', 'w') as f:
        json.dump(best_config_dict, f, indent=4)
    print(f"✓ Best config saved to: {save_dir}/best_config.json\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 70)
    
    plot_hyperparameter_heatmaps(results_df, save_dir)
    plot_parameter_effects(results_df, save_dir)
    plot_top_configurations(results_df, top_n=5, save_dir=save_dir)
    
    # Create table of top configurations
    create_top_configurations_table(results_df, top_n=5, save_dir=save_dir)
    
    # Compare top 3 configurations with learning curves
    print("\nGenerating learning curve comparisons (this may take a while)...")
    top_3 = results_df.nlargest(3, 'success_rate_mean')
    top_configs = []
    for _, row in top_3.iterrows():
        config = {
            'alpha': row['alpha'],
            'gamma': row['gamma'],
            'epsilon_start': row['epsilon_start'],
            'epsilon_min': row['epsilon_min'],
            'epsilon_decay': row['epsilon_decay']
        }
        top_configs.append(config)
    
    plot_learning_curves_comparison(top_configs, num_episodes=10000, 
                                   max_steps=100, save_dir=save_dir)
    
    print("\n" + "="*70)
    print("TUNING COMPLETE!")
    print(f"All results and plots saved to: {save_dir}/")
    print("="*70 + "\n")
    
    return results_df, best_config_dict


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run tuning
    results_df, best_config = main()
