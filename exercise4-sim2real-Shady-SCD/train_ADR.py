import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.core import Wrapper

# Ensure this import matches your file structure
from env.custom_hopper import *

# --- CONFIGURATION ---
LOG_DIR = "./training_logs_dr/"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTEPS = 100000 

# --- ADVANCED DR WRAPPER ---
class AdvancedDRWrapper(Wrapper):
    """
    Advanced Domain Randomization Wrapper supporting:
    - Uniform
    - Gaussian (Normal)
    - Skewed (Beta)
    - Biased means (Shifted Center)
    """
    def __init__(self, env, dr_type='uniform', range_scale=0.2, shift_factor=0.0):
        """
        Args:
            dr_type (str): 'uniform', 'gaussian', or 'skewed'.
            range_scale (float): Magnitude of randomization (e.g., 0.2 = +/- 20%).
            shift_factor (float): Bias to add to the mean (e.g., +0.1 means center is 10% heavier).
                                  Used for 'Gaussian centered on target' logic.
        """
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
            
            # Apply Bias (Simulating a guess that target is heavier/lighter)
            center_mass = nominal * (1.0 + self.shift_factor)
            
            # Calculate bounds based on the CENTER, not the original nominal
            delta = nominal * self.range_scale # Keep width relative to nominal size
            low = center_mass - delta
            high = center_mass + delta
            
            # --- SAMPLING LOGIC ---
            if self.dr_type == 'uniform':
                # Standard UDR
                val = np.random.uniform(low, high)
                
            elif self.dr_type == 'gaussian':
                # Gaussian: Mean = center, Std = delta / 3 (so 99% falls within range)
                sigma = delta / 3.0
                val = np.random.normal(loc=center_mass, scale=sigma)
                
            elif self.dr_type == 'skewed':
                # Beta Distribution: Skew towards one side
                # alpha=2, beta=5 -> Skews Left (Lower masses favored)
                # alpha=5, beta=2 -> Skews Right (Heavier masses favored)
                # We use alpha=5, beta=2 to bias towards heavier (Target is usually heavier in these labs)
                val = np.random.beta(5, 2) 
                # Map [0,1] output of Beta to [low, high]
                val = low + val * (high - low)
                
            else:
                val = nominal # Fallback

            # Safety clip (mass cannot be negative)
            new_masses[i] = max(0.01, val)

        self.unwrapped.model.body_mass[:] = new_masses
        return self.env.reset(**kwargs)

# --- TRAINING HELPER ---
def train_experiment(env_name, exp_name, dr_config):
    """
    Trains a single experiment configuration.
    dr_config: dict containing 'dr_type', 'range_scale', 'shift_factor'
    """
    print(f"--- Starting Experiment: {exp_name} ---")
    print(f"    Config: {dr_config}")
    
    env = gym.make(env_name)
    
    # Apply Advanced DR Wrapper
    env = AdvancedDRWrapper(env, **dr_config)
    
    # Logging
    log_path = os.path.join(LOG_DIR, exp_name)
    env = Monitor(env, filename=log_path)
    
    # Train
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(os.path.join(LOG_DIR, f"model_{exp_name}"))
    env.close()
    return model

# --- PLOTTING ---
def plot_results(experiments):
    plt.figure(figsize=(12, 7))
    
    for name, _ in experiments.items():
        path = os.path.join(LOG_DIR, name + ".monitor.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, skiprows=1)
            # Smooth data
            y = df['r'].rolling(window=200).mean()
            x = np.cumsum(df['l'])
            plt.plot(x, y, label=name)
            
    plt.title("Stability Analysis: DR Distribution Types")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("dr_stability_analysis.png")
    print("Saved plot to dr_stability_analysis.png")

def main():
    # Define Experiments
    # We assume 'CustomHopper-source-v0' is the base env to randomize
    base_env = 'CustomHopper-source-v0'
    
    # 1. Define the comparison configurations
    experiments = {
        # Baseline
        "1_Uniform_Standard":  {"dr_type": "uniform",  "range_scale": 0.5, "shift_factor": 0.0},
        
        # Distribution comparison
        "2_Gaussian_Centered": {"dr_type": "gaussian", "range_scale": 0.5, "shift_factor": 0.0},
        
        # Range comparison
        "3_Uniform_Narrow":    {"dr_type": "uniform",  "range_scale": 0.1, "shift_factor": 0.0},
        "4_Uniform_Wide":      {"dr_type": "uniform",  "range_scale": 0.8, "shift_factor": 0.0},
        
        # Biased / Skewed
        # Assuming Target is heavier -> Shift center by +30%
        "5_Gaussian_Biased":   {"dr_type": "gaussian", "range_scale": 0.3, "shift_factor": 0.3},
        "6_Skewed_Beta":       {"dr_type": "skewed",   "range_scale": 0.5, "shift_factor": 0.0},
    }

    # 2. Run Training Loop
    models = {}
    for exp_name, config in experiments.items():
        models[exp_name] = train_experiment(base_env, exp_name, config)

    # 3. Plot Training Curves (Stability Analysis)
    plot_results(experiments)

    # 4. Final Sim-to-Real Evaluation
    print("\n=== SIM-TO-REAL TRANSFER RESULTS (Evaluated on TARGET) ===")
    results = {}
    target_env_name = 'CustomHopper-target-v0' 
    
    for exp_name in experiments.keys():
        model_path = os.path.join(LOG_DIR, f"model_{exp_name}")
        
        # Create fresh target env for eval
        eval_env = gym.make(target_env_name)
        model = PPO.load(model_path)
        
        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
        results[exp_name] = mean_r
        print(f"{exp_name:25}: {mean_r:.2f} +/- {std_r:.2f}")
        eval_env.close()

    # 5. Interpretability Bar Chart
    plt.figure(figsize=(10, 6))
    plt.barh(list(results.keys()), list(results.values()), color='skyblue')
    plt.axvline(x=results["1_Uniform_Standard"], color='r', linestyle='--', label='Uniform Baseline')
    plt.title("Effect of DR Strategy on Target Environment Performance")
    plt.xlabel("Mean Reward on Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dr_transfer_comparison.png")

if __name__ == "__main__":
    main()