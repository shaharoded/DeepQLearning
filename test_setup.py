"""
Quick test script to verify installation and agent functionality.

Run this after installing dependencies to ensure everything works correctly.
"""

import sys
import numpy as np

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found. Install with: pip install torch")
        return False
    
    try:
        import gymnasium as gym
        print(f"✓ Gymnasium {gym.__version__}")
    except ImportError:
        print("✗ Gymnasium not found. Install with: pip install gymnasium")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib")
    except ImportError:
        print("✗ Matplotlib not found. Install with: pip install matplotlib")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False
    
    return True


def test_agents():
    """Test that all agent classes can be instantiated."""
    print("\nTesting agent classes...")
    
    try:
        from src.agent import QLearningAgent, DeepQLearningAgent, ImprovedDeepQLearningAgent
        
        # Test basic instantiation
        agent1 = QLearningAgent(state_dim=100, action_dim=4)
        print("✓ QLearningAgent")
        
        agent2 = DeepQLearningAgent(state_dim=4, action_dim=2)
        print("✓ DeepQLearningAgent")
        
        agent3 = ImprovedDeepQLearningAgent(state_dim=4, action_dim=2)
        print("✓ ImprovedDeepQLearningAgent")
        
        return True
        
    except Exception as e:
        print(f"✗ Error instantiating agents: {e}")
        return False


def test_environment():
    """Test that Gymnasium environments work."""
    print("\nTesting environment...")
    
    try:
        import gymnasium as gym
        
        env = gym.make('CartPole-v1')
        state, _ = env.reset()
        
        print(f"✓ CartPole-v1 environment")
        print(f"  State shape: {state.shape}")
        print(f"  Action space: {env.action_space}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error with environment: {e}")
        return False


def test_basic_training():
    """Test a very short training run."""
    print("\nTesting basic training (5 episodes)...")
    
    try:
        import gymnasium as gym
        from src.agent import DeepQLearningAgent
        
        env = gym.make('CartPole-v1')
        
        config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_capacity': 1000
        }
        
        agent = DeepQLearningAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config
        )
        
        # Very short training
        metrics = agent.train(env, num_episodes=5, verbose=False)
        
        print(f"✓ Training completed")
        print(f"  Episodes: {len(metrics['rewards'])}")
        print(f"  Mean reward: {np.mean(metrics['rewards']):.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING INSTALLATION TESTS")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test agents
    if results[-1][1]:  # Only if imports succeeded
        results.append(("Agent Classes", test_agents()))
        results.append(("Environment", test_environment()))
        results.append(("Basic Training", test_basic_training()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    print("="*60)
    
    if all(passed for _, passed in results):
        print("\n🎉 All tests passed! You're ready to start training agents.")
        print("\nNext steps:")
        print("  1. Run 'python main.py' to train an example agent")
        print("  2. Check API_DESIGN.md for detailed API documentation")
        print("  3. Modify the code to experiment with different configurations")
        return True
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
