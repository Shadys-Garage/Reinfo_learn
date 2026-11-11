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

from env.custom_hopper import *

# --- CONFIGURATION ---
LOG_DIR = "./training_logs_batch_complete/"
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
                # INTELLIGENT SKEW:
                # If shifting POSITIVE (Heavy), skew towards the right (Beta 5,2)
                # If shifting NEGATIVE (Light), skew towards the left (Beta 2,5)
                if self.shift_factor >= 0:
                    val = np.random.beta(5, 2) 
                else:
                    val = np.random.beta(2, 5)
                
                val = low + val * (high - low)
                
            else:
                val = nominal 

            # Safety clip
            new_masses[i] = max(0.01, val)
        
        self.unwrapped.model.body_mass[:] = new_masses
        return self.env.reset(**kwargs)

# --- BATCH CONFIG GENERATOR ---
def get_batch_configs():
    experiments = []
    
    # 1. Uniform
    ranges = [0.1, 0.3, 0.5, 0.7, 1.0]
    for r in ranges:
        experiments.append({
            "name": f"Uniform_Range{r}",
            "config": {"dr_type": "uniform", "range_scale": r, "shift_factor": 0.0},
            "group": "Uniform",
            "sort_val": r
        })

    # 2. Gaussian
    for r in ranges:
        experiments.append({
            "name": f"Gaussian_Range{r}",
            "config": {"dr_type": "gaussian", "range_scale": r, "shift_factor": 0.0},
            "group": "Gaussian",
            "sort_val": r
        })

    # 3. Skewed Heavy (Positive Shifts)
    shifts_pos = [0.2, 0.4, 0.6]
    for s in shifts_pos:
        experiments.append({
            "name": f"Skewed_Heavy_Shift+{s}",
            "config": {"dr_type": "skewed", "range_scale": 0.5, "shift_factor": s},
            "group": "Skewed_Heavy",
            "sort_val": s
        })

    # 4. Skewed Light (Negative Shifts) - NEW!
    shifts_neg = [-0.2, -0.4, -0.6]
    for s in shifts_neg:
        experiments.append({
            "name": f"Skewed_Light_Shift{s}",
            "config": {"dr_type": "skewed", "range_scale": 0.5, "shift_factor": s},
            "group": "Skewed_Light",
            "sort_val": abs(s) # Use absolute value for color intensity
        })
        
    return experiments

# --- PLOTTING HELPER: COLORS ---
def get_color_for_exp(group, val):
    cmaps = {
        "Uniform": plt.cm.Blues,
        "Gaussian": plt.cm.Reds,
        "Skewed_Heavy": plt.cm.Purples,
        "Skewed_Light": plt.cm.Greens  # New Green map for Light Skews
    }
    
    if "Skewed" in group:
        intensity = 0.4 + val  # 0.2->0.6 maps to 0.6->1.0
    else:
        intensity = 0.3 + (val * 0.6) 
        
    return cmaps[group](intensity)

# --- PLOTTING: DISTRIBUTIONS ---
def plot_all_distributions(experiments):
    plt.figure(figsize=(14, 8))
    x = np.linspace(0.0, 3.0, 1000)
    
    for exp in experiments:
        conf = exp["config"]
        color = get_color_for_exp(exp["group"], exp["sort_val"])
        
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
            with np.errstate(divide='ignore', invalid='ignore'):
                # Flip Beta parameters based on shift
                if conf['shift_factor'] >= 0:
                    y = stats.beta.pdf(z, 5, 2) / (high - low)
                else:
                    y = stats.beta.pdf(z, 2, 5) / (high - low)
            y = np.where((x >= low) & (x <= high), y, 0)
            
        plt.plot(x, y, color=color, alpha=0.8, linewidth=2, label=exp["name"])

    plt.title("Distribution Showdown: Heavy vs Light vs Neutral")
    plt.xlabel("Mass Multiplier (1.0 = Nominal)")
    plt.ylabel("Probability Density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "all_distributions.png"))

# --- PLOTTING: LEARNING CURVES ---
def plot_all_learning_curves(experiments):
    plt.figure(figsize=(14, 8))
    
    for exp in experiments:
        path = os.path.join(LOG_DIR, exp["name"] + ".monitor.csv")
        if not os.path.exists(path): continue
            
        try:
            df = pd.read_csv(path, skiprows=1)
            y = df['r'].rolling(window=200).mean()
            x = np.cumsum(df['l'])
            color = get_color_for_exp(exp["group"], exp["sort_val"])
            plt.plot(x, y, color=color, linewidth=2, label=exp["name"])
        except: pass

    plt.title("Learning Curves: Complete Comparison")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "all_learning_curves.png"))

# --- PLOTTING: FINAL BAR CHART ---
def plot_final_comparison(results_data):
    plt.figure(figsize=(16, 8)) # Wider to fit more bars
    
    # Sort by Mean Reward
    results_data.sort(key=lambda x: x['mean'], reverse=True)
    
    names = [r['name'] for r in results_data]
    means = [r['mean'] for r in results_data]
    stds = [r['std'] for r in results_data]
    colors = [get_color_for_exp(r['group'], r['sort_val']) for r in results_data]
    
    plt.bar(names, means, yerr=stds, color=colors, capsize=5, alpha=0.9, edgecolor='black')
    
    plt.title("Final Sim-to-Real Performance (Target Environment)")
    plt.ylabel("Mean Reward +/- Std Dev")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(LOG_DIR, "final_performance_bar_chart.png"))
    print(f"-> Saved Final Bar Chart to {LOG_DIR}")

# --- TRAINING ---
def train_config(env_name, exp):
    print(f"\n--- Training {exp['name']} ---")
    env = gym.make(env_name)
    env = AdvancedDRWrapper(env, **exp['config'])
    env = Monitor(env, filename=os.path.join(LOG_DIR, exp["name"]))
    
    model = PPO("MlpPolicy", env, verbose=0, device='cpu')
    model.learn(total_timesteps=TIMESTEPS)
    model.save(os.path.join(LOG_DIR, f"model_{exp['name']}"))
    env.close()
    return model

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='batch')
    args = parser.parse_args()

    if args.mode == 'batch':
        experiments = get_batch_configs()
        
        # 1. Distributions Plot
        plot_all_distributions(experiments)
        
        # 2. Train & Evaluate
        results_data = [] 
        eval_env = gym.make('CustomHopper-target-v0')
        
        for exp in experiments:
            model = train_config('CustomHopper-source-v0', exp)
            
            # Evaluate
            mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20)
            print(f"   -> Result on Target: {mean_r:.2f} +/- {std_r:.2f}")
            
            results_data.append({
                "name": exp["name"],
                "mean": mean_r,
                "std": std_r,
                "group": exp["group"],
                "sort_val": exp["sort_val"]
            })

        eval_env.close()
        
        # 3. Plot Curves & Bars
        plot_all_learning_curves(experiments)
        plot_final_comparison(results_data)
        
        # 4. CSV
        df = pd.DataFrame(results_data)
        df = df[['name', 'mean', 'std']]
        df.sort_values(by='mean', ascending=False, inplace=True)
        df.to_csv(os.path.join(LOG_DIR, "final_results.csv"), index=False)

    else:
        print("Run with --mode batch")