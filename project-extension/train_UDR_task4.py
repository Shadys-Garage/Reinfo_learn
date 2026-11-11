import numpy as np
import gymnasium as gym
import argparse
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

class UDRWrapper(gym.Wrapper):
    def __init__(self, env, randomization_range=0.3):
        """
        randomization_range: factor to vary the masses (e.g., 0.3 means +/- 30%)
        """
        super().__init__(env)
        self.randomization_range = randomization_range
        self.base_masses = np.copy(self.env.unwrapped.model.body_mass) #Original Masses
        
    def reset(self, **kwargs):
        new_masses = np.copy(self.base_masses)        
        # Randomize masses for Thigh (2), Leg (3), and Foot (4)        
        for i in [2, 3, 4]:
            low = self.base_masses[i] * (1 - self.randomization_range)
            high = self.base_masses[i] * (1 + self.randomization_range)
            new_masses[i] = np.random.uniform(low, high)

        # Apply the randomized masses to the MuJoCo model
        self.env.unwrapped.model.body_mass[:] = new_masses        
        return self.env.reset(**kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--net_size", type=int, default=64, help="Neurons per layer")
    parser.add_argument("--range", type=float, default=0.3, help="UDR randomization range")
    return parser.parse_args()

def main():
    args = parse_args()
    policy_kwargs = dict(net_arch=[args.net_size, args.net_size])
    # Create the environment with UDR
    base_env = gym.make("CustomHopper-source-v0")
    env_udr = UDRWrapper(base_env, randomization_range=args.range)
    
    # Evaluation environments (Without randomization)
    env_source_clean = gym.make("CustomHopper-source-v0")
    env_target = gym.make("CustomHopper-target-v0")

    # Train PPO on the UDR environment
    print(f"Starting UDR training (Range: {args.range})...")
    model = PPO("MlpPolicy", env_udr, learning_rate=args.lr, gamma=args.gamma, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=args.steps)

    # Final Evaluation
    print(f"\n--- UDR Results (Range: {args.range}) ---")
    mean_s, std_s = evaluate_policy(model, env_source_clean, n_eval_episodes=50)
    mean_t, std_t = evaluate_policy(model, env_target, n_eval_episodes=50)
    
    print(f"UDR Agent -> Source: {mean_s:.2f} +/- {std_s:.2f}")
    print(f"UDR Agent -> Target: {mean_t:.2f} +/- {std_t:.2f}")
    
    model.save(f"ppo_hopper_udr_{args.range}")

if __name__ == '__main__':
    main()