"""
Script for training Control Policies with Configurable Domain Randomization.
Usage Examples:
    python train_UDR_argparse.py --dr_type gaussian --range_scale 0.5
    python train_UDR_argparse.py --dr_type uniform --range_scale 0.2
"""
import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import scipy.stats as stats  # For plotting theoretical PDFs
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.core import Wrapper

from env.custom_hopper import *

# --- CONFIGURATION ---
LOG_DIR = "./training_logs_dr_args/"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTEPS = 100000 

# --- ADVANCED DR WRAPPER ---
class AdvancedDRWrapper(Wrapper):
    """
    Wrapper that applies domain randomization based on user config.
    """
    def __init__(self, env, dr_type='uniform', range_scale=0.2, shift_factor=0.0):
        super().__init__(env)
        self.dr_type = dr_type
        self.range_scale = range_scale
        self.shift_factor = shift_factor
        
        # Original nominal masses from Source
        self.original_masses = np.copy(self.unwrapped.model.body_mass)
        self.random_indices = [2, 3, 4] # Thigh, Leg, Foot

    def reset(self, **kwargs):
        new_masses = np.copy(self.original_masses)
        
        for i in self.random_indices:
            nominal = self.original_masses[i]
            
            # 1. Apply Shift (Bias)
            center_mass = nominal * (1.0 + self.shift_factor)
            
            # 2. Calculate Bounds
            delta = nominal * self.range_scale 
            low = center_mass - delta
            high = center_mass + delta
            
            # 3. Sample
            if self.dr_type == 'uniform':
                val = np.random.uniform(low, high)
                
            elif self.dr_type == 'gaussian':
                # Sigma is set so that 3 standard deviations = range_scale
                sigma = delta / 3.0
                val = np.random.normal(loc=center_mass, scale=sigma)
                
            elif self.dr_type == 'skewed':
                # Beta(5,2) biases towards the heavier side (right skew)
                val = np.random.beta(5, 2)
                val = low + val * (high - low)
                
            else:
                val = nominal 

            # Clip to ensure positive mass
            new_masses[i] = max(0.01, val)

        self.unwrapped.model.body_mass[:] = new_masses
        return self.env.reset(**kwargs)

# --- VISUALIZATION HELPER ---
def visualize_dr_distribution(dr_type, range_scale, shift_factor, save_path="current_dist_plot.png"):
    """
    Plots the theoretical shape of the mass distribution being used.
    Normalized to a nominal mass of 1.0 for clarity.
    """
    plt.figure(figsize=(8, 5))
    
    nominal = 1.0
    center = nominal * (1.0 + shift_factor)
    delta = nominal * range_scale
    low = center - delta
    high = center + delta
    
    # Create an x-axis range
    x = np.linspace(center - delta*1.5, center + delta*1.5, 500)
    
    if dr_type == 'uniform':
        # Uniform PDF: 1/(high-low) inside bounds, 0 outside
        y = np.where((x >= low) & (x <= high), 1/(high-low), 0)
        title = f"Uniform Distribution (Range +/- {range_scale*100}%)"
        color = 'blue'
        
    elif dr_type == 'gaussian':
        sigma = delta / 3.0
        y = stats.norm.pdf(x, center, sigma)
        title = f"Gaussian Distribution (Sigma ~ {range_scale*100/3:.1f}%)"
        color = 'red'
        
    elif dr_type == 'skewed':
        # Beta is defined on [0, 1], so we must map x back to [0, 1]
        # z = (x - low) / (high - low)
        z = (x - low) / (high - low)
        y = stats.beta.pdf(z, 5, 2) / (high - low) # Normalize pdf
        y = np.where((x >= low) & (x <= high), y, 0)
        title = f"Skewed Beta(5,2) (Biased Heavy)"
        color = 'purple'
    
    else:
        y = np.zeros_like(x)
        title = "Unknown"
        color = 'black'

    # Plotting
    plt.plot(x, y, color=color, lw=3, label='Probability Density')
    plt.fill_between(x, y, color=color, alpha=0.3)
    
    # Mark the Nominal (1.0)
    plt.axvline(1.0, color='black', linestyle='--', label='Original Mass (1.0)')
    if shift_factor != 0:
        plt.axvline(center, color='green', linestyle=':', label='Shifted Center')

    plt.title(title)
    plt.xlabel("Mass Multiplier (1.0 = Nominal)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    print(f"-> Distribution plot saved to: {save_path}")
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Train Hopper with Configurable Domain Randomization")
    
    parser.add_argument('--dr_type', type=str, default='gaussian', 
                        choices=['uniform', 'gaussian', 'skewed'],
                        help="Type of distribution: uniform, gaussian, or skewed")
    
    parser.add_argument('--range_scale', type=float, default=0.5,
                        help="How wide the randomization is (e.g., 0.5 = +/- 50%)")
    
    parser.add_argument('--shift_factor', type=float, default=0.0,
                        help="Bias shift (e.g., 0.2 = center is 20% heavier)")

    args = parser.parse_args()

    # 2. Visualize the chosen distribution BEFORE training
    print(f"--- Configuration: Type={args.dr_type}, Scale={args.range_scale}, Shift={args.shift_factor} ---")
    plot_filename = f"dist_{args.dr_type}_scale{args.range_scale}.png"
    visualize_dr_distribution(args.dr_type, args.range_scale, args.shift_factor, 
                              save_path=os.path.join(LOG_DIR, plot_filename))

    # 3. Setup Environment
    env_name = 'CustomHopper-source-v0'
    env = gym.make(env_name)
    
    # Wrap with our configurable wrapper
    env = AdvancedDRWrapper(env, 
                            dr_type=args.dr_type, 
                            range_scale=args.range_scale, 
                            shift_factor=args.shift_factor)
    
    # Logging
    exp_name = f"{args.dr_type}_scale{args.range_scale}"
    env = Monitor(env, filename=os.path.join(LOG_DIR, exp_name))

    # 4. Train
    print("-> Starting Training...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save Model
    model_path = os.path.join(LOG_DIR, f"model_{exp_name}")
    model.save(model_path)
    print(f"-> Model saved to {model_path}")

    # 5. Plot Learning Curve (Simple)
    try:
        df = pd.read_csv(os.path.join(LOG_DIR, exp_name + ".monitor.csv"), skiprows=1)
        plt.figure()
        plt.plot(np.cumsum(df['l']), df['r'].rolling(100).mean())
        plt.title(f"Learning Curve: {exp_name}")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(LOG_DIR, f"curve_{exp_name}.png"))
        print(f"-> Learning curve saved to curve_{exp_name}.png")
    except Exception as e:
        print("Could not plot learning curve (maybe too few steps?):", e)