import gymnasium as gym
from env.custom_hopper import *
import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Hopper with Plotting")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for PPO")
    parser.add_argument("--net_size", type=int, default=64, help="Neurons per layer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Environment selection
    parser.add_argument("--train_env", type=str, default="CustomHopper-source-v0", 
                        choices=["CustomHopper-source-v0", "CustomHopper-target-v0"], 
                        help="Environment to train on")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory for saved models")
    parser.add_argument("--plot_name", type=str, default="learning_curve.png", help="Name of the output plot file")
    
    return parser.parse_args()

def plot_results(log_folder, title="Learning Curve", output_file="learning_curve.png"):
    """
    Reads the monitor.csv files from stable-baselines3 and plots the smoothed reward.
    """
    x, y = [], []
    
    # SB3 Monitor logs are often split into multiple files if parallelized, 
    # but here we likely have one. We look for all *.monitor.csv files in the dir.
    monitor_files = glob.glob(os.path.join(log_folder, "*.monitor.csv"))
    
    if not monitor_files:
        print("No monitor files found. Skipping plotting.")
        return

    data_frames = []
    for file in monitor_files:
        # standard SB3 monitor files have 2 header lines
        df = pd.read_csv(file, skiprows=1)
        data_frames.append(df)
    
    # Concatenate if there are multiple logs (e.g. from previous runs in same dir)
    # Ideally, clear the log dir before a fresh run to avoid mixing curves.
    full_df = pd.concat(data_frames, axis=0)
    
    # 'r' is reward, 't' is time (elapsed), 'l' is episode length.
    # We plot rolling mean to smooth the noise
    window_size = 50
    rolling_mean = full_df['r'].rolling(window=window_size).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(full_df['r'], alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(rolling_mean, color='blue', linewidth=2, label=f'Moving Avg ({window_size} eps)')
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    save_path = os.path.join(log_folder, output_file)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show() # Show in notebook/window if available

def main():
    args = parse_args()
    
    # Ensure directories exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    run_name = f"ppo_{args.train_env.split('-')[1]}_s{args.steps}_sd{args.seed}"
    
    # Initialize Environments
    # We create a specific sub-folder for logs to keep them clean for plotting
    run_log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    train_env = gym.make(args.train_env)
    # Monitor is crucial here for the CSV logs needed for plotting
    train_env = Monitor(train_env, filename=os.path.join(run_log_dir, "monitor"))
    
    source_env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0")

    print(f"--- Training Configuration: {run_name} ---")
    
    policy_kwargs = dict(net_arch=[dict(pi=[args.net_size, args.net_size], vf=[args.net_size, args.net_size])])

    model = PPO(
        "MlpPolicy", 
        train_env, 
        learning_rate=args.lr, 
        gamma=args.gamma,
        batch_size=args.batch_size,
        policy_kwargs=policy_kwargs, 
        seed=args.seed,
        verbose=1
    )

    eval_callback = EvalCallback(
        train_env, 
        best_model_save_path=run_log_dir,
        log_path=run_log_dir, 
        eval_freq=5000, 
        deterministic=True, 
        render=False
    )

    print(f"\nStarting training...")
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    print("Training completed.")

    # --- PLOTTING ---
    print("\nGenerating Learning Curve...")
    plot_results(run_log_dir, title=f"Learning Curve: {args.train_env}", output_file="learning_curve.png")

    # --- EVALUATION ---
    print(f"\n--- Final Evaluation Results ---")
    
    # Load best model for evaluation
    best_model_path = os.path.join(run_log_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        model = PPO.load(best_model_path)
        print("Loaded best model for evaluation.")
    
    mean_s, std_s = evaluate_policy(model, source_env, n_eval_episodes=50, deterministic=True)
    print(f"Policy -> Source: {mean_s:.2f} +/- {std_s:.2f}")

    mean_t, std_t = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)
    print(f"Policy -> Target: {mean_t:.2f} +/- {std_t:.2f}")

    # Save final model
    model.save(os.path.join(args.model_dir, run_name))

if __name__ == '__main__':
    main()