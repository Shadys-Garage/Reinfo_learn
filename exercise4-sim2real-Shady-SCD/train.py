"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


from env.custom_hopper import *

LOG_DIR = "./training_logs/"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTEPS = 200000  # Adjust as needed (e.g., 100000 for quick test, 1000000 for convergence)

def train_agent(env_name, model_name, log_name):
    """
    Trains an agent and saves learning logs for plotting.
    """
    print(f"--- Training on {env_name} ---")
    
    # CHANGE 1: Create environment
    env = gym.make(env_name)
    
    # CHANGE 2: Wrap the environment with Monitor to log data (reward, time, etc.) to a CSV file
    # This is crucial for plotting the learning curves later.
    env = Monitor(env, filename=os.path.join(LOG_DIR, log_name))
    
    # Initialize PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save
    model.save(model_name)
    env.close()
    return model

def plot_learning_curves(log_names):
    """
    Reads the monitor.csv files generated during training and plots the learning curves.
    """
    plt.figure(figsize=(10, 6))
    
    for label, filename in log_names.items():
        path = os.path.join(LOG_DIR, filename + ".monitor.csv")
        if not os.path.exists(path):
            print(f"Warning: Log file {path} not found.")
            continue
            
        # Read the logs. Skip the first 1 line (metadata)
        try:
            df = pd.read_csv(path, skiprows=1)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        # Calculate rolling mean for smoother plots (window of 50 episodes)
        # 'r' is the column for rewards, 'l' is for episode length
        window_size = 100
        rolling_mean = df['r'].rolling(window=window_size).mean()
        
        # Plot
        # We use the number of episodes (index) or cumsum of steps (l) for X-axis
        x_axis = np.cumsum(df['l'])
        plt.plot(x_axis, rolling_mean, label=f"{label} (Moving Avg)")

    plt.title("Learning Curve: Source vs Target Training")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve_comparison.png")
    print("Plot saved as 'learning_curve_comparison.png'")
    # plt.show() # Uncomment if you want to see the window pop up

def render_final_policy(model_path, env_name):
    """
    Loads a model and renders it visually.
    """
    print(f"--- Rendering model {model_path} on {env_name} ---")
    print("Press Ctrl+C in the terminal to stop rendering if it doesn't close automatically.")
    
    # CHANGE 3: Enable rendering by setting render_mode='human'
    env = gym.make(env_name, render_mode='human')
    
    model = PPO.load(model_path)
    
    # Run for a few episodes to visualize
    episodes_to_watch = 5
    for ep in range(episodes_to_watch):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # We must use deterministic=True for evaluation to see the best behavior
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Note: env.render() is handled automatically by render_mode='human' in newer gym versions
            # If using older gym, you might need explicit env.render() here.
            
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
    
    env.close()

def evaluate_performance(model_path, env_name):
    """
    Standard numerical evaluation (no rendering).
    """
    env = gym.make(env_name)
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    env.close()
    return mean_reward, std_reward
def main():
   # env = gym.make('CustomHopper-source-v0')

   # print('State space:', env.observation_space)  # state-space
   # print('Action space:', env.action_space)  # action-space
   # print('Dynamics parameters:', env.unwrapped.get_parameters())  # masses of each link of the Hopper

    """
        TODO:

            - train a policy with stable-baselines3 on the source Hopper env
            - test the policy with stable-baselines3 on <source,target> Hopper envs (hint: see the evaluate_policy method of stable-baselines3)
    """
    # 1. Train Source
    train_agent('CustomHopper-source-v0', 'ppo_source', 'log_source')
    
    # 2. Train Target (for comparison)
    train_agent('CustomHopper-target-v0', 'ppo_target', 'log_target')
    
    # 3. Plot the training progress (Rewards vs Steps)
    plot_learning_curves({
        'Source Training': 'log_source',
        'Target Training': 'log_target'
    })
    
    # 4. Numerical Evaluation (Task 2.3)
    print("\n=== FINAL RESULTS ===")
    results = {}
    
    # Source -> Source
    m, s = evaluate_performance('ppo_source', 'CustomHopper-source-v0')
    results['S->S'] = m
    print(f"Source -> Source: {m:.2f} +/- {s:.2f}")

    # Source -> Target (Lower Bound)
    m, s = evaluate_performance('ppo_source', 'CustomHopper-target-v0')
    results['S->T'] = m
    print(f"Source -> Target: {m:.2f} +/- {s:.2f}")

    # Target -> Target (Upper Bound)
    m, s = evaluate_performance('ppo_target', 'CustomHopper-target-v0')
    results['T->T'] = m
    print(f"Target -> Target: {m:.2f} +/- {s:.2f}")
    
    # Plot Bar Chart for Final Results
    plt.figure()
    plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green'])
    plt.title("Sim-to-Real Transfer Performance")
    plt.ylabel("Mean Reward")
    plt.savefig("transfer_performance.png")
    print("Performance bar chart saved as 'transfer_performance.png'")

    # 5. RENDER THE FINAL TEST
    # Ask user if they want to render
    user_input = input("\nDo you want to visualize the 'Source -> Target' transfer? (y/n): ")
    if user_input.lower() == 'y':
        # This solves the user's request to "render the final test"
        render_final_policy('ppo_source', 'CustomHopper-target-v0')
if __name__ == '__main__':
    main()