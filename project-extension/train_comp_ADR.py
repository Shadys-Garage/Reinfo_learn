"""
Script for comparing User-Defined DR Strategy vs. Standard UDR Baseline.
Usage Examples:
    # Compare Gaussian (Narrow) vs Standard UDR
    python train_compare_UDR.py --dr_type gaussian --range_scale 0.1

    # Compare Skewed (Biased) vs Standard UDR
    python train_compare_UDR.py --dr_type skewed --range_scale 0.5 --shift_factor 0.3
"""
import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import scipy.stats as stats
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.core import Wrapper

# Ensure this matches your file structure
from env.custom_hopper import *

# --- CONFIGURATION ---
LOG_DIR = "./training_logs_compare/"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTEPS = 100000 

# --- ADVANCED DR WRAPPER (Same as before) ---
class AdvancedDRWrapper(Wrapper):
    def __init__(self, env, dr_type='uniform', range_scale=0.2, shift_factor=0.0):
        super().__init__(env)
        self.dr_type = dr_type
        self.range_scale = range_scale
        self.shift_factor = shift_factor
        self.original_masses = np.copy(self.unwrapped.model.body_mass)
        self.random_indices = [2, 3, 4] 

    def reset(self, **kwargs):
        new_masses = np.copy(self.original_masses)
        for i in self.random_indices:
            nominal = self.original_masses[i]
            center_mass = nominal * (1.0 + self.shift_factor)
            delta = nominal * self.range_scale 
            low = center_mass - delta
            high = center_mass + delta
            
            if self.dr_type == 'uniform':
                val = np.random.uniform(low, high)
            elif self.dr_type == 'gaussian':
                sigma = delta / 3.0
                val = np.random.normal(loc=center_mass, scale=sigma)
            elif self.dr_type == 'skewed':
                # Beta(5,2) biases towards heavier side
                val = np.random.beta(5, 2)
                val = low + val * (high - low)
            else:
                val = nominal 

            new_masses[i] = max(0.01, val)
        
        self.unwrapped.model.body_mass[:] = new_masses
        return self.env.reset(**kwargs)

# --- VISUALIZATION HELPER ---
def visualize_distributions(baseline_conf, user_conf, save_path="dist_compare.png"):
    """
    Plots BOTH the Baseline and the User distribution on one chart for comparison.
    """
    plt.figure(figsize=(10, 6))
    x = np.linspace(0.0, 2.5, 500) # Range relative to nominal 1.0
    
    configs = [("Standard UDR", baseline_conf, 'blue'), ("Your Strategy", user_conf, 'orange')]
    
    for label, conf, color in configs:
        nominal = 1.0
        center = nominal * (1.0 + conf['shift_factor'])
        delta = nominal * conf['range_scale']
        low = center - delta
        high = center + delta
        
        if conf['dr_type'] == 'uniform':
            y = np.where((x >= low) & (x <= high), 1/(high-low), 0)
        elif conf['dr_type'] == 'gaussian':
            sigma = delta / 3.0
            y = stats.norm.pdf(x, center, sigma)
        elif conf['dr_type'] == 'skewed':
            z = (x - low) / (high - low)
            y = stats.beta.pdf(z, 5, 2) / (high - low)
            y = np.where((x >= low) & (x <= high), y, 0)
        
        plt.plot(x, y, color=color, lw=3, label=f"{label} ({conf['dr_type']})")
        plt.fill_between(x, y, color=color, alpha=0.2)

    plt.axvline(1.0, color='black', linestyle='--', label='Original Mass')
    plt.title("Distribution Showdown: Baseline vs. Experiment")
    plt.xlabel("Mass Multiplier")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"-> Distribution comparison saved to: {save_path}")
    plt.close()

# --- TRAINING MANAGER ---
def train_config(env_name, config, log_name):
    """
    Trains a model with a specific DR configuration.
    """
    print(f"\n--- Training {log_name} ---")
    print(f"    Config: {config}")
    
    env = gym.make(env_name)
    env = AdvancedDRWrapper(env, **config)
    env = Monitor(env, filename=os.path.join(LOG_DIR, log_name))
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(os.path.join(LOG_DIR, f"model_{log_name}"))
    env.close()
    return model

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dr_type', type=str, default='gaussian', choices=['uniform', 'gaussian', 'skewed'])
    parser.add_argument('--range_scale', type=float, default=0.5)
    parser.add_argument('--shift_factor', type=float, default=0.0)
    args = parser.parse_args()

    # 1. Define Configurations
    baseline_config = {
        'dr_type': 'uniform', 
        'range_scale': args.range_scale, 
        'shift_factor': 0.0
    }
    
    user_config = {
        'dr_type': args.dr_type,
        'range_scale': args.range_scale,
        'shift_factor': args.shift_factor
    }

    # 2. Visualize Comparison
    visualize_distributions(baseline_config, user_config, save_path=os.path.join(LOG_DIR, "dist_comparison.png"))

    # 3. Train Both
    # We check if baseline exists to save time, otherwise train it
    baseline_model_path = os.path.join(LOG_DIR, "model_baseline_udr.zip")
    if os.path.exists(baseline_model_path):
        print("-> Baseline UDR model found. Loading...")
        model_baseline = PPO.load(baseline_model_path)
    else:
        model_baseline = train_config('CustomHopper-source-v0', baseline_config, "baseline_udr")

    # Always train the user experiment fresh
    model_user = train_config('CustomHopper-source-v0', user_config, "user_experiment")

    # 4. Compare Learning Curves
    plt.figure(figsize=(10, 6))
    
    # Plot Baseline
    try:
        df_base = pd.read_csv(os.path.join(LOG_DIR, "baseline_udr.monitor.csv"), skiprows=1)
        plt.plot(np.cumsum(df_base['l']), df_base['r'].rolling(100).mean(), label="Baseline (Standard UDR)", color='blue', linestyle='--')
    except: pass
    
    # Plot User
    try:
        df_user = pd.read_csv(os.path.join(LOG_DIR, "user_experiment.monitor.csv"), skiprows=1)
        plt.plot(np.cumsum(df_user['l']), df_user['r'].rolling(100).mean(), label="Your Experiment", color='orange', linewidth=2)
    except: pass
    
    plt.title(f"Training Stability: Standard UDR vs {args.dr_type}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, "comparison_learning_curve.png"))
    print("-> Learning curve comparison saved.")

    # 5. Final Evaluation (Sim-to-Real)
    print("\n=== FINAL HEAD-TO-HEAD RESULTS (Target Env) ===")
    eval_env = gym.make('CustomHopper-target-v0')
    
    # Evaluate Baseline
    mean_b, std_b = evaluate_policy(model_baseline, eval_env, n_eval_episodes=20)
    print(f"Standard UDR:    {mean_b:.2f} +/- {std_b:.2f}")
    
    # Evaluate User
    mean_u, std_u = evaluate_policy(model_user, eval_env, n_eval_episodes=20)
    print(f"Your Strategy:   {mean_u:.2f} +/- {std_u:.2f}")
    
    # Result
    diff = mean_u - mean_b
    if diff > 0:
        print(f"RESULT: Your strategy IMPROVED performance by +{diff:.2f} points!")
    else:
        print(f"RESULT: Your strategy performed WORSE by {diff:.2f} points.")
    
    eval_env.close()
    