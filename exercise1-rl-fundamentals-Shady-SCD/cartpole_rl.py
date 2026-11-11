"""
    Robot Learning
    Exercise 1

    Reinforcement Learning 

    Polito A-Y 2025-2026
"""
import torch
import gymnasium as gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import scipy
import sys
from agent import Agent, Policy
from utils import get_space_dim

import sys


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", type=float, default=0.0,
                        help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--multiplelr", action='store_true', 
                        help="Train with multiple learning rates and plot comparison")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0, x0=0, random_policy=False):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation, _ = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1
                """
                Sample a random action from the action space
                """
                # TODO
                action = env.action_space.sample() #this ignores the agent policy and uses the environment's random action
               # Create uniform probabilities for all actions as PyTorch tensor
                action_probabilities = torch.ones(env.action_space.n) / env.action_space.n
                # Make sure it's on the same device as the agent
                action_probabilities = action_probabilities.to(agent.train_device)

            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # Task 3.1: optionally overwrite the original reward with a shaped reward
            if x0 is not None:
                try:
                    # new_reward expects (state, x0) and returns a scalar
                    reward = new_reward(observation, x0)
                except Exception:
                    # fallback to original reward if shaping fails
                    pass

            # TODO
            # reward = new_reward(observation, x0)
            if x0 is not None:
                try:
                    reward = new_reward(observation, x0)
                except Exception:
                    pass

            # Store action's outcome (so that the agent can improve its policy)ù
            if not random_policy:
                agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                # Gymnasium recommends setting render_mode at env creation
                pass

            # Store total episode reward
            reward_sum += reward
            timesteps += 1
        
        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps : 
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        if not random_policy:
            agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False, x0=0):
    test_reward, test_len = 0, 0

    episodes = 100
    print('Num testing episodes:', episodes)

    if render:
        plt.ion()
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
        
        axL.set_title('Position & Velocity')
        axL.set_xlabel('Timestep')
        axL.set_ylabel('Value')
        axL.grid(True, alpha=0.3)
        
        axR.set_title('Angle & Angular Velocity') 
        axR.set_xlabel('Timestep')
        axR.set_ylabel('Value')
        axR.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
    
    for ep in range(episodes):
        done = False
        observation, _ = env.reset()
        old_state = observation

        if render:
            states_history = []
            timesteps = []
            time = 0

        while not done:
        # Task 1.3
            """
            Test on 500 timesteps
            """
            # TODO

            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            # reward = new_reward(observation, x0)

            if render:
                states_history.append(observation.copy())
                timesteps.append(time)
                time += 1
                
                axL.clear()
                axR.clear()
                
                states_array = np.array(states_history)
                timesteps_array = np.array(timesteps)
                
                axL.plot(timesteps_array, states_array[:, 0], 'b-', label='Position', linewidth=1)
                axL.plot(timesteps_array, states_array[:, 1], 'r-', label='Velocity', linewidth=1)
                axL.set_title('Position & Velocity')
                axL.legend()
                axL.grid(True, alpha=0.3)
                
                axR.plot(timesteps_array, states_array[:, 2], 'g-', label='Angle', linewidth=1)
                axR.plot(timesteps_array, states_array[:, 3], 'm-', label='Angular Velocity', linewidth=1)
                axR.set_title('Angle & Angular Velocity')
                axR.legend()
                axR.grid(True, alpha=0.3)
                
                if int(time) > 100:
                    axL.set_xlim(time - 100, time + 10)
                    axR.set_xlim(time - 100, time + 10)
                
                plt.pause(1e-3)
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)



def new_reward(state, x0):
    """
    Task 3.1: Custom reward functions
    Choose which reward function to use by uncommenting one of the lines below
    """
    # Uncomment ONE of the following lines to use that reward function:
    
    #return balance_at_center(state, x0)      # Reward function 1
    #return balance_at_point(state, x0)       # Reward function 2  
    return move_back_forth(state, x0)         # Reward function 3
    #return 1

def balance_at_center(state, x0):
    """
    Reward function 1: Balance pole close to center x=0
    
    """
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
    
    # Reward for keeping pole upright (primary objective)
    angle_reward = 1.0 - abs(pole_angle) / 0.4  # Normalize by ~12 degrees (0.2 radians)
    
    # Penalty for being away from center
    position_penalty = 0.5 * abs(cart_position/2.4)
    
    # Combined reward (prioritize angle, but encourage center position)
    reward = max(angle_reward - position_penalty, 0.1)  # Minimum reward of 0.1
    
    return reward

def balance_at_point(state, x0):
    """
    Reward function 2: Balance pole at arbitrary point x=x0
    """
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
    
    survival_reward = 1
    angle_reward= 0.4-0.4*abs(pole_angle)
    speed_bonus = 0.8 * (min(abs(cart_velocity), 2.0)-cart_position)  # Cap speed bonus
    dist_from_point = max(abs(x0-cart_position),0.1)
    #if abs(cart_position) > x0:
    position_penality = 0.2*abs((cart_position)**2-(abs(x0-0.2)))

    position_reward = 0.2*(1-exp(abs(cart_position-x0)))
    #else:
    #    position_penality = 0
    reward = max(survival_reward+ angle_reward + speed_bonus + position_reward -position_penality, 0.1)
    return reward

def move_back_forth(state, x0):
    """
    Reward function 3: Move cart back and forth while balancing pole
    """
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
    
    survival_reward = 1
    distance_from_x0 = abs(cart_position - x0)
    # Base reward for balancing
    angle_reward= 2 *abs(pole_angle)
    speed_bonus = 2 * min(abs(cart_velocity), 2.0)  # Cap speed bonus
    # Bonus when reaching far points (encourage turning around)
    turning_bonus =1 - 1/exp(distance_from_x0)
    
    # Bonus for changing direction near boundaries
    if abs(cart_position - x0) > 1.5:  # Far from x0 (encourage turning back)
        boundary_penality = 0.6
    else:
        boundary_penality = 0
    if abs(cart_velocity) < 0.4:  # Far from x0 (encourage turning back)
        stagnation_penality = 0.3
    else:
        stagnation_penality = 0.0
    
    reward = max(survival_reward+ angle_reward + speed_bonus + turning_bonus - stagnation_penality - boundary_penality, 0.1)

    return reward

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    render_mode = "human" if (args.render_training or args.render_test) else None
    env = gym.make(args.env, render_mode=render_mode, max_episode_steps=500)

    # Set seed for reproducible random policy
    if args.random_policy:
        env.reset(seed=42)
        env.action_space.seed(42)
    # Task 1.3
    
    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    if args.multiplelr:
        learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        print("Training with multiple learning rates:", learning_rates)
        print("="*60)
        
        plt.figure(figsize=(12, 8))
        
        for lr, color in zip(learning_rates, colors):
            print(f"\nTraining with learning rate: {lr}")
            
            # Create new environment for each learning rate
            env_lr = gym.make(args.env, render_mode=render_mode,max_episode_steps=200)
            
            # Instantiate agent and its policy with current learning rate
            policy = Policy(observation_space_dim, action_space_dim)
            agent = Agent(policy, lr=lr)
            
            # Train with current learning rate
            training_history = train(agent, env_lr, args.train_episodes, False, 
                                   args.render_training, x0=args.central_point, 
                                   random_policy=args.random_policy)
            
            # Plot the reward history for this learning rate
            plt.plot(training_history['episode'], training_history['reward'], 
                    label=f'LR: {lr}', color=color, linewidth=2)
            
            # Save the model for this learning rate
            model_file = "%s_lr_%s_params.ai" % (args.env, str(lr))
            torch.save(policy.state_dict(), model_file)
            print(f"Model saved to {model_file}")
            
            env_lr.close()
        
        # Customize the comparison plot
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Task 1.3: Learning Rate Comparison\nReward vs Episode for Different Learning Rates')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\nLearning rate comparison plot saved as 'learning_rate_comparison.png'")
        
    else:
        # Instantiate agent and its policy
        policy = Policy(observation_space_dim, action_space_dim)
        agent = Agent(policy, lr=args.lr)

        # Print some stuff
        print("Environment:", args.env)
        print("Training device:", agent.train_device)
        print("Observation space dimensions:", observation_space_dim)
        print("Action space dimensions:", action_space_dim)

        # If no model was passed, train a policy from scratch.
        # Otherwise load the policy from the file and go directly to testing.
        if args.test is None:
            # Train
            training_history = train(agent, env, args.train_episodes, False, args.render_training, x0=args.central_point, random_policy=args.random_policy)

            # Save the model
            model_file = "%s_params.ai" % args.env
            torch.save(policy.state_dict(), model_file)
            print("Model saved to", model_file)

            # Plot rewards
            matplotlib.use('TkAgg')
            sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
            sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
            plt.legend()
            plt.title("Reward history (%s)" % args.env)
            plt.show()
            print("Training finished.")
        else:
            # Test
            print("Loading model from", args.test, "...")
            state_dict = torch.load(args.test)
            policy.load_state_dict(state_dict)
            print("Testing...")
            test(agent, env, args.train_episodes, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

