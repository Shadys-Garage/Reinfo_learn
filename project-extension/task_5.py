import numpy as np
import gymnasium as gym
import argparse
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

class AblativeUDRWrapper(gym.Wrapper):
    def __init__(self, env, randomization_range=0.3, component="all"):
        """
        component: "thigh", "leg", "foot", or "all"
        """
        super().__init__(env)
        self.randomization_range = randomization_range
        self.component = component
        self.base_masses = np.copy(self.env.unwrapped.model.body_mass)
        
        # Mappatura componenti -> indici MuJoCo
        # 2: Thigh, 3: Leg, 4: Foot
        self.comp_map = {
            "thigh": [2],
            "leg": [3],
            "foot": [4],
            "all": [2, 3, 4]
        }
        
    def reset(self, **kwargs):
        new_masses = np.copy(self.base_masses)
        indices_to_randomize = self.comp_map.get(self.component, [2, 3, 4])
        
        for i in indices_to_randomize:
            low = self.base_masses[i] * (1 - self.randomization_range)
            high = self.base_masses[i] * (1 + self.randomization_range)
            new_masses[i] = np.random.uniform(low, high)

        # Applichiamo le masse (il torso all'indice 1 rimane sempre fisso)
        self.env.unwrapped.model.body_mass[:] = new_masses        
        return self.env.reset(**kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="Ablative UDR Study")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--net_size", type=int, default=64)
    parser.add_argument("--range", type=float, default=0.3)
    # --- NUOVO ARGOMENTO PER L'ESTENSIONE ---
    parser.add_argument("--component", type=str, default="all", 
                        choices=["all", "thigh", "leg", "foot"], 
                        help="Which component to randomize")
    return parser.parse_args()

def main():
    args = parse_args()
    policy_kwargs = dict(net_arch=[args.net_size, args.net_size])

    # Inizializziamo l'ambiente con il componente specifico
    base_env = gym.make("CustomHopper-source-v0")
    env_udr = AblativeUDRWrapper(base_env, randomization_range=args.range, component=args.component)
    
    env_source_clean = gym.make("CustomHopper-source-v0")
    env_target = gym.make("CustomHopper-target-v0")

    print(f"\n--- Running Extension: Ablative UDR ---")
    print(f"Randomizing: {args.component.upper()} | Range: {args.range}")

    model = PPO("MlpPolicy", env_udr, learning_rate=args.lr, gamma=args.gamma, 
                policy_kwargs=policy_kwargs, verbose=1)
    
    model.learn(total_timesteps=args.steps)

    # Valutazione
    mean_s, std_s = evaluate_policy(model, env_source_clean, n_eval_episodes=50)
    mean_t, std_t = evaluate_policy(model, env_target, n_eval_episodes=50)
    
    print(f"\nResults for {args.component}:")
    print(f"Source: {mean_s:.2f} +/- {std_s:.2f}")
    print(f"Target: {mean_t:.2f} +/- {std_t:.2f}")
    
    model.save(f"ppo_ablative_{args.component}")

if __name__ == '__main__':
    main()