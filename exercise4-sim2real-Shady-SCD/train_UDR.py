"""
Script for training Control Policies (Standard & UDR) on Hopper.
Solves Exercises 2.2, 2.3, and 3 (Task 4).
"""
import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.core import Wrapper

from env.custom_hopper import *

# --- CONFIGURATION ---
LOG_DIR = "./training_logs/"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTEPS = 200000  # Keep consistent for fair comparison

# --- NEW: UDR WRAPPER (Task 4) ---
class UDRHopperWrapper(Wrapper):
    """
    A wrapper that performs Uniform Domain Randomization (UDR) on the Hopper's masses
    at the beginning of every episode.
    """
    def __init__(self, env, mass_range_scale=0.2):
        """
        mass_range_scale: Percentage of variation (e.g., 0.2 means +/- 20% variation)
        """
        super().__init__(env)
        self.mass_range_scale = mass_range_scale
        
        # Get the original masses (nominal values from Source)
        # We access the underlying MuJoCo model
        self.original_masses = np.copy(self.unwrapped.model.body_mass)
        
        # Indices to randomize: 2 (thigh), 3 (leg), 4 (foot)
        # We DO NOT randomize index 1 (torso) as per instructions.
        self.random_indices = [2, 3, 4] 

    def reset(self, **kwargs):
        """
        Resets the environment and samples new dynamics parameters.
        """
        # 1. Sample new masses
        new_masses = np.copy(self.original_masses)
        
        for i in self.random_indices:
            nominal_mass = self.original_masses[i]
            
            # Uniform distribution: [nominal * (1-scale), nominal * (1+scale)]
            low = nominal_mass * (1.0 - self.mass_range_scale)
            high = nominal_mass * (1.0 + self.mass_range_scale)
            
            new_masses[i] = np.random.uniform(low, high)

        # 2. Apply new masses to the MuJoCo model
        # Note: In newer Gymnasium/MuJoCo versions, we might need to be careful 
        # to ensure these changes take effect immediately.
        self.unwrapped.model.body_mass[:] = new_masses

        # 3. Print debug info (optional, helps verify UDR is working)
        # print(f"DEBUG UDR: Masses set to {new_masses[1:5]}") 

        return self.env.reset(**kwargs)

# --- TRAINING HELPER ---
def train_agent(env_name, model_name, log_name, use_udr=False):
    """
    Trains an agent. Supports optional UDR wrapping.
    """
    label = "UDR" if use_udr else "Standard"
    print(f"--- Training {label} Agent on {env_name} ---")
    
    # 1. Create Environment
    env = gym.make(env_name)
    
    # 2. Apply UDR Wrapper if requested
    if use_udr:
        # We randomize by +/- 50% here to ensure we cover the "Target" domain gap.
        # You can tune 'mass_range_scale' as a hyperparameter.
        env = UDRHopperWrapper(env, mass_range_scale=0.5)
        print("-> UDR Wrapper Applied: Randomizing masses [2,3,4] by +/- 50%")
    
    # 3. Wrap for logging
    env = Monitor(env, filename=os.path.join(LOG_DIR, log_name))
    
    # 4. Train
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)
    
    model.save(model_name)
    env.close()
    return model

# --- PLOTTING HELPER ---
def plot_learning_curves(log_names):
    plt.figure(figsize=(10, 6))
    for label, filename in log_names.items():
        path = os.path.join(LOG_DIR, filename + ".monitor.csv")
        if not os.path.exists(path): continue
        
        try:
            df = pd.read_csv(path, skiprows=1)
            # Rolling mean for smoothness
            rolling_mean = df['r'].rolling(window=100).mean()
            x_axis = np.cumsum(df['l'])
            plt.plot(x_axis, rolling_mean, label=label)
        except: pass

    plt.title("Learning Curves: Source vs Target vs UDR")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve_comparison.png")
    print("Plot saved as 'learning_curve_comparison.png'")

# --- EVALUATION HELPER ---
def evaluate_performance(model_path, env_name):
    env = gym.make(env_name)
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    env.close()
    return mean_reward, std_reward

def main():
    # --- PHASE 1: Standard Training (Already done? You can comment these out if you have the models) ---
    # train_agent('CustomHopper-source-v0', 'ppo_source', 'log_source', use_udr=False)
    # train_agent('CustomHopper-target-v0', 'ppo_target', 'log_target', use_udr=False)
    
    # --- PHASE 2: UDR Training (Task 4) ---
    # We train on Source, but with the UDR wrapper active
    train_agent('CustomHopper-source-v0', 'ppo_udr', 'log_udr', use_udr=True)

    # --- PHASE 3: Plotting ---
    plot_learning_curves({
        'Source (Standard)': 'log_source',
        'Target (Upper Bound)': 'log_target',
        'Source (UDR)': 'log_udr'
    })
    
    # --- PHASE 4: Evaluation & Comparison ---
    print("\n=== FINAL SIM-TO-REAL RESULTS ===")
    results = {}
    
    # 1. Baseline: Source -> Target (Lower Bound)
    m, s = evaluate_performance('ppo_source', 'CustomHopper-target-v0')
    results['Standard S->T'] = m
    print(f"Standard Source -> Target: {m:.2f} +/- {s:.2f}")
    
    # 2. UDR: Source(UDR) -> Target (Our Solution)
    m, s = evaluate_performance('ppo_udr', 'CustomHopper-target-v0')
    results['UDR S->T'] = m
    print(f"UDR Source -> Target:      {m:.2f} +/- {s:.2f}")

    # 3. Oracle: Target -> Target (Upper Bound)
    m, s = evaluate_performance('ppo_target', 'CustomHopper-target-v0')
    results['Oracle T->T'] = m
    print(f"Target -> Target:          {m:.2f} +/- {s:.2f}")

    # Plot Comparison
    plt.figure()
    colors = ['orange', 'purple', 'green']
    plt.bar(results.keys(), results.values(), color=colors)
    plt.title("Impact of UDR on Sim-to-Real Transfer")
    plt.ylabel("Mean Reward on Target Env")
    plt.savefig("udr_comparison.png")
    print("Comparison bar chart saved as 'udr_comparison.png'")

if __name__ == '__main__':
    main()