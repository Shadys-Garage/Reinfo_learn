# Modified cartpole.py to compare different REINFORCE variants

import torch
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train_reinforce(env_name, variant='basic', train_run_id=0, train_episodes=1000):
    """
    variant: 'basic', 'with_baseline', 'normalized'
    """
    env = gym.make(env_name)
    action_space_dim = env.action_space.shape[-1]
    observation_space_dim = env.observation_space.shape[-1]
    
    print(f'Training REINFORCE {variant}...')
    print(f'action_space_dim: {action_space_dim}')
    print(f'observation_space_dim: {observation_space_dim}')

    policy = Policy(observation_space_dim, action_space_dim)
    
    # Configure agent based on variant
    if variant == 'basic':
        agent = Agent(policy, use_baseline=False, normalize_rewards=False)
    elif variant == 'with_baseline':
        agent = Agent(policy, use_baseline=True, normalize_rewards=False)
    elif variant == 'normalized':
        agent = Agent(policy, use_baseline=True, normalize_rewards=True)
    
    reward_history, timestep_history = [], []
    average_reward_history = []

    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        observation, _ = env.reset()

        while not done:
            action, action_probabilities = agent.get_action(observation, episode_number=episode_number)
            previous_observation = observation
            
            observation, reward, terminated, truncated, info = env.step(action.detach().numpy())
            done = terminated or truncated
            
            agent.store_outcome(previous_observation, action_probabilities, action, reward)
            reward_sum += reward
            timesteps += 1

        if episode_number % 100 == 0:
            print(f"Episode {episode_number}: Total reward: {reward_sum:.1f} ({timesteps} timesteps)")

        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        agent.episode_finished(episode_number)

    # Save results
    data = pd.DataFrame({
        "episode": np.arange(len(reward_history)),
        "train_run_id": [train_run_id] * len(reward_history),
        "algorithm": [f"REINFORCE_{variant}"] * len(reward_history),
        "reward": reward_history,
        "avg_reward": average_reward_history
    })
    
    torch.save(agent.policy.state_dict(), f"model_{env_name}_{variant}_{train_run_id}.mdl")
    return data


def compare_variants(env_name, train_episodes=1000):
    """Compare different REINFORCE variants"""
    variants = ['basic', 'with_baseline', 'normalized']
    results = []
    
    for variant in variants:
        print(f"\n{'='*50}")
        print(f"Training {variant}")
        print('='*50)
        
        data = train_reinforce(env_name, variant, train_episodes=train_episodes)
        data['variant'] = variant
        results.append(data)
        
        # Plot individual variant
        plt.figure(figsize=(10, 6))
        plt.plot(data['reward'], alpha=0.3, label='Episode reward')
        plt.plot(data['avg_reward'], linewidth=2, label='100-episode average')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'REINFORCE - {variant}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'reinforce_{variant}.png')
        plt.show()
    
    # Combine and plot all variants
    all_results = pd.concat(results)
    
    plt.figure(figsize=(12, 8))
    for variant in variants:
        variant_data = all_results[all_results['variant'] == variant]
        plt.plot(variant_data['avg_reward'], label=f'REINFORCE {variant}', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100-episode window)')
    plt.title('Comparison of REINFORCE Variants')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('reinforce_comparison.png')
    plt.show()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--variant", type=str, default="all", choices=["basic", "with_baseline", "normalized", "all"], 
                       help="REINFORCE variant to use")
    parser.add_argument("--compare", action='store_true', help="Compare all variants")
    args = parser.parse_args()
    
    if args.compare or args.variant == "all":
        compare_variants(args.env, args.train_episodes)
    elif args.test is None:
        train_reinforce(args.env, args.variant, train_episodes=args.train_episodes)
    else:
        # Testing code (similar to original)
        env = gym.make(args.env)
        action_space_dim = env.action_space.shape[-1]
        observation_space_dim = env.observation_space.shape[-1]
        
        policy = Policy(observation_space_dim, action_space_dim)
        policy.load_state_dict(torch.load(args.test))
        agent = Agent(policy)
        
        test_rewards = []
        for ep in range(100):
            done = False
            observation, _ = env.reset()
            total_reward = 0
            
            while not done:
                action, _ = agent.get_action(observation, evaluation=True)
                observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
                done = terminated or truncated
                total_reward += reward
            
            test_rewards.append(total_reward)
        
        print(f"Average test reward: {np.mean(test_rewards):.2f} +/- {np.std(test_rewards):.2f}")