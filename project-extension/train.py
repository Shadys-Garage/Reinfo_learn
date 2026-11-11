"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gymnasium as gym
from env.custom_hopper import *
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Hopper Environment")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--net_size", type=int, default=64, help="Neurons per layer")
    parser.add_argument("--train_env", type=str, default="CustomHopper-source-v0", choices=["CustomHopper-source-v0", "CustomHopper-target-v0"], help="Environment to train on")
    # Evaluation parameters
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of episodes for evaluation")
    
    return parser.parse_args()

def main():
    args = parse_args()
    policy_kwargs = dict(net_arch=[args.net_size, args.net_size])
    # Initialize enviroments
    train_env = gym.make(args.train_env)
    source_env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0")

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.unwrapped.get_parameters())  # masses of each link of the Hopper

    model = PPO("MlpPolicy", train_env, learning_rate=args.lr, gamma=args.gamma,policy_kwargs=policy_kwargs , verbose=1)

    # Training:
    print(f"\nStarting training for {args.steps} steps...")
    model.learn(total_timesteps=args.steps)
    print("Training completed.")

    print(f"\n--- Results for Model trained on {args.train_env} ---")


    # Evaluation, Training Env -> Source:    
    mean_s, std_s = evaluate_policy(model, source_env, n_eval_episodes=args.eval_episodes)
    print(f"{args.train_env} -> Source: {mean_s:.2f} +/- {std_s:.2f}")

    # Evaluation, Training Env -> Target:
    mean_t, std_t = evaluate_policy(model, target_env, n_eval_episodes=args.eval_episodes)
    print(f"{args.train_env} -> Target: {mean_t:.2f} +/- {std_t:.2f}")

    # Save model with a descriptive name
    model_name = f"ppo_{args.train_env.split('-')[1]}_g{args.gamma}_lr{args.lr}"
    model.save(model_name)
if __name__ == '__main__':
    main()