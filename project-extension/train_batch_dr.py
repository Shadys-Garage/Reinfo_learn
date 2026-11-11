import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
LOG_DIR = "./training_logs_batch/"
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTEPS = 200000 

# --- ADVANCED DR WRAPPER ---
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
                sigma = delta / 3.0
                val = np.random.normal(loc=center_mass, scale=sigma)
            elif self.dr_type == 'skewed':
                # Beta(5,2) biases towards heavier side (right skew)
                val = np.random.beta(5, 2)
                val = low + val * (high - low)
            else:
                val = nominal 

            new_masses[i] = max(0.01, val)
        
        self.unwrapped.model.body_mass[:] = new_masses
        return self.env.reset(**kwargs)

# --- BATCH CONFIG GENERATOR ---
def get_batch_configs():
    """Generates the list of all experiments to run."""
    experiments = []
    
    # 1. Uniform (Varying Ranges: Narrow -> Wide)
    ranges = [0.1, 0.3, 0.5, 0.7, 1.0]
    for r in ranges:
        experiments.append({
            "name": f"Uniform_Range{r}",
            "config": {"dr_type": "uniform", "range_scale": r, "shift_factor": 0.0},
            "group": "Uniform",
            "color_val": r  # For shading
        })

    # 2. Gaussian (Varying Ranges: Narrow -> Wide)
    for r in ranges:
        experiments.append({
            "name": f"Gaussian_Range{r}",
            "config": {"dr_type": "gaussian", "range_scale": r, "shift_factor": 0.0},
            "group": "Gaussian",
            "color_val": r
        })

    # 3. Skewed (Varying Shifts: 0.2, 0.4, 0.6)
    # We fix range at 0.5 for these to isolate the shift effect
    shifts = [0.2, 0.4, 0.6]
    for s in shifts:
        experiments.append({
            "name": f"Skewed_Shift{s}",
            "config": {"dr_type": "skewed", "range_scale": 0.5, "shift_factor": s},
            "group": "Skewed",
            "color_val": s
        })
        
    return experiments

# --- PLOTTING: DISTRIBUTIONS ---
def plot_all_distributions(experiments):
    plt.figure(figsize=(14, 8))
    x = np.linspace(0.0, 3.0, 1000)
    
    # Colormaps for groups
    cmaps = {
        "Uniform": plt.cm.Blues,
        "Gaussian": plt.cm.Reds,
        "Skewed": plt.cm.Purples
    }
    
    for exp in experiments:
        conf = exp["config"]
        group = exp["group"]
        
        # Color logic: map value to color intensity
        if group == "Skewed":
            # Shifts 0.2->0.6 map to color 0.5->0.9
            intensity = 0.4 + conf["shift_factor"]
        else:
            # Ranges 0.1->1.0 map to color 0.4->1.0
            intensity = 0.3 + (conf["range_scale"] * 0.6)
            
        color = cmaps[group](intensity)
        
        # Distribution Calculation
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
            # Handle out of bounds for beta
            with np.errstate(divide='ignore', invalid='ignore'):
                y = stats.beta.pdf(z, 5, 2) / (high - low)
            y = np.where((x >= low) & (x <= high), y, 0)
            
        plt.plot(x, y, color=color, alpha=0.8, linewidth=2, label=exp["name"])

    plt.title("Distribution Showdown: All Configurations")
    plt.xlabel("Mass Multiplier (1.0 = Nominal)")
    plt.ylabel("Probability Density")
    plt.axvline(1.0, color='black', linestyle='--', alpha=0.5)
    
    # Place legend outside to avoid clutter
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(LOG_DIR, "all_distributions.png"))
    print("-> Saved distribution plot: all_distributions.png")

# --- PLOTTING: LEARNING CURVES ---
def plot_all_learning_curves(experiments):
    plt.figure(figsize=(14, 8))
    
    cmaps = {
        "Uniform": plt.cm.Blues,
        "Gaussian": plt.cm.Reds,
        "Skewed": plt.cm.Purples
    }
    
    for exp in experiments:
        path = os.path.join(LOG_DIR, exp["name"] + ".monitor.csv")
        if not os.path.exists(path):
            continue
            
        try:
            df = pd.read_csv(path, skiprows=1)
            # Smooth
            y = df['r'].rolling(window=200).mean()
            x = np.cumsum(df['l'])
            
            # Reconstruct color
            conf = exp["config"]
            group = exp["group"]
            if group == "Skewed":
                intensity = 0.4 + conf["shift_factor"]
            else:
                intensity = 0.3 + (conf["range_scale"] * 0.6)
            color = cmaps[group](intensity)
            
            plt.plot(x, y, color=color, linewidth=2, label=exp["name"])
        except:
            pass

    plt.title("Learning Curves: Mega-Comparison")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "all_learning_curves.png"))
    print("-> Saved learning curves: all_learning_curves.png")

# --- TRAINING ---
def train_config(env_name, exp):
    print(f"\n--- Training {exp['name']} ---")
    env = gym.make(env_name)
    env = AdvancedDRWrapper(env, **exp['config'])
    env = Monitor(env, filename=os.path.join(LOG_DIR, exp["name"]))
    
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=TIMESTEPS)
    model.save(os.path.join(LOG_DIR, f"model_{exp['name']}"))
    env.close()
    return model

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='batch', choices=['batch', 'single'],
                        help="Run 'batch' for all experiments, or 'single' for one specific config.")
    # Arguments for Single Mode
    parser.add_argument('--dr_type', type=str, default='gaussian')
    parser.add_argument('--range_scale', type=float, default=0.5)
    parser.add_argument('--shift_factor', type=float, default=0.0)
    args = parser.parse_args()

    if args.mode == 'batch':
        # 1. Get all configs
        experiments = get_batch_configs()
        
        # 2. Visualize Distributions First
        plot_all_distributions(experiments)
        
        # 3. Run Training Loop
        results = {}
        eval_env = gym.make('CustomHopper-target-v0')
        
        for exp in experiments:
            # Train
            model = train_config('CustomHopper-source-v0', exp)
            
            # Evaluate on Target immediately
            mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
            results[exp['name']] = mean_r
            print(f"   -> Result on Target: {mean_r:.2f} +/- {std_r:.2f}")

        eval_env.close()
        
        # 4. Plot Curves
        plot_all_learning_curves(experiments)
        
        # 5. Save Final Text Report
        res_df = pd.DataFrame.from_dict(results, orient='index', columns=['Mean Reward'])
        res_df.sort_values(by='Mean Reward', ascending=False, inplace=True)
        print("\n=== FINAL RANKING ===")
        print(res_df)
        res_df.to_csv(os.path.join(LOG_DIR, "final_results.csv"))

    else:
        # Run Single Mode (Legacy Support)
        print("Running Single Config Mode...")
        # (You can just call the train_config with the args here if needed)
        pass