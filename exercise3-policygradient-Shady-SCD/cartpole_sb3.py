import torch
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd

import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

# T4 TODO
def create_model(args, env):
    if args.algo.lower() == 'ppo':
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            device="cpu",
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.98,
            verbose=1,
            seed=args.seed,
            tensorboard_log="./tmp/tb"
        )

    elif args.algo.lower() == 'sac':
        grad_steps = args.gradient_steps if args.gradient_steps > 0 else 1
        model = SAC(
            policy="MlpPolicy",
            env=env,
            device="cpu",
            learning_rate=args.lr,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            gamma=0.98,
            gradient_steps=grad_steps,
            verbose=1,
            seed=args.seed,
            tensorboard_log="./tmp/tb"
        )
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    
    return model

# T4 TODO
def load_model(args, env):
    if args.algo.lower() == 'ppo':
        return PPO.load(args.test, env=env, device='cpu')
    elif args.algo.lower() == 'sac':
        return SAC.load(args.test, env=env, device='cpu')
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=25000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--test_episodes', default=100, type=int, help='# episodes for test evaluations')
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make(args.env)

    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)


    if args.test is None:
        try:
            model = create_model(args, env)
            # Policy training (T4)
            print("Starting training...")
            import time
            start_time = time.time()
            model.learn(total_timesteps=args.total_timesteps)
            train_time = time.time() - start_time
            print(f"Training finished in {train_time:.2f} seconds.")
            
            # Saving model (T4) TODO
            save_path = f"./{args.algo}_final"
            model.save(save_path)
            print(f"Saved trained model to: {save_path}")

            # Plot the learning curve
            plot_results(log_dir)

        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
            try:
                model.save(f"./{args.algo}_partial")
                print("Saved partial model.")
            except:
                print("Could not save partial model.")


    else:
        print("Testing...")
        model = load_model(args, env)

        # Policy evaluation (T4)
        mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.test_episodes,
        deterministic=True,
        render=args.render_test
    )


        print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")

    env.close()    

