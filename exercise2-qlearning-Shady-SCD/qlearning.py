import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd

import sys

np.random.seed(123)

env = gym.make('CartPole-v1')

# Whether to perform training or use the stored .npy file
MODE = 'TRAINING' # TRAINING, TEST

episodes = 20000
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0.2
b = 2300  # TODO: choose b so that with GLIE we get an epsilon of 0.1 after 20'000 episodes

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
#q_grid = np.zeros((discr, discr, discr, discr, num_of_actions))

# Option (b) - Optimistic initialization  
q_grid = np.full((discr, discr, discr, discr, num_of_actions), 50.0)

if MODE == 'TEST':
    q_grid = np.load('q_values.npy')

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False):
    x, v, th, av = get_cell_index(state)

    if greedy: # TEST -> greedy policy
        best_action_estimated = np.argmax(q_grid[x, v, th, av, :]) # TODO: greedy w.r.t. q_grid

        return best_action_estimated

    else: # TRAINING -> epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Random action
            action_chosen = np.random.randint(0, num_of_actions)  # TODO: choose random action with equal probability among all actions

            return action_chosen
        else:
            # Greedy action
            best_action_estimated = np.argmax(q_grid[x, v, th, av, :])  # TODO: greedy w.r.t. q_grid

            return best_action_estimated


def update_q_value(old_state, action, new_state, reward, done, q_array):
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        target_value = reward + gamma * np.max(q_grid[new_cell_index[0], new_cell_index[1], new_cell_index[2], new_cell_index[3], :])  # TODO

    # Update Q value
    old_q = q_grid[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action]
    q_grid[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = old_q + alpha * (target_value - old_q)
    #q_grid[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = ...  # TODO

    return

def compute_value_function(): #additional start #new
    """Compute value function V(s) = max_a Q(s,a) for Task 3.2"""
    value_function = np.max(q_grid, axis=4)
    return value_function

def plot_heatmap(value_function, title="Value Function Heatmap"):#new
    """Plot heatmap averaged over velocity and angular velocity for Task 3.2"""
    # Average over velocity (v) and angular velocity (av) dimensions
    v_avg = np.mean(value_function, axis=(1, 3))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(v_avg, cmap='viridis', 
                xticklabels=np.round(th_grid, 2), 
                yticklabels=np.round(x_grid, 2))
    plt.xlabel('Pole Angle (θ)')
    plt.ylabel('Cart Position (x)')
    plt.title(title)
    plt.show()
    
    return v_avg #additional end

# Training loop
ep_lengths, epl_avg = [], []
returns = []  # For tracking returns per episode #addiotional aprt
epsilon_values = []  # For tracking epsilon over time #addiotional aprt
#new
for ep in range(episodes+test_episodes):
    test = ep > episodes

    if MODE == 'TEST':
        test = True

    state, _ = env.reset(seed=321)
    done = False
    steps = 0
    episode_return = 0

    # Task 3.1: Epsilon schedule - uncomment the desired one
    
    # Option 1: Constant epsilon
    #epsilon = constant_eps
    
    # Option 2: GLIE schedule
    #epsilon = b / (b + ep) if not test else 0.0
    
    # Option 3: For Task 3.3 - set epsilon to 0
    epsilon = 0.0

    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
        
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            pass  # env.render()  # Uncomment to visualize

        state = new_state
        steps += 1

    ep_lengths.append(steps)
    returns.append(episode_return)
    epsilon_values.append(epsilon)
    epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))
    
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}, return: {:.1f}".format(
            ep, np.mean(ep_lengths[max(0, ep - 200):]), episode_return))
        print('Epsilon:', epsilon)
#new

# Plot training progress for Task 3.1 #additional start
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(returns)#error
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Returns during Training')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epsilon_values[:episodes])
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Schedule (GLIE)')
plt.grid(True)

plt.tight_layout()
plt.show() 
# Task 3.2: Compute and plot value function heatmap
if MODE == 'TRAINING':
    value_func = compute_value_function()
    heatmap_data = plot_heatmap(value_func, "Learned Value Function Heatmap")
    
    # Save the Q-value array
    np.save("q_values.npy", q_grid)
if MODE == 'TEST':
    # For testing, load and analyze the trained Q-values
    value_func = compute_value_function()
    plot_heatmap(value_func, "Trained Value Function Heatmap")
    sys.exit() #additional end
    #sys.exit() 

# Save the Q-value array
np.save("q_values.npy", q_grid)