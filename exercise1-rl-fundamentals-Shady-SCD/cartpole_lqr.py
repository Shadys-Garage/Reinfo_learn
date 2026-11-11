"""
    Robot Learning
    Exercise 1

    Linear Quadratic Regulator

    Polito A-Y 2025-2026
"""
import gymnasium as gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import matplotlib
import sys
from utils import get_space_dim, set_seed
import pdb 
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    # state matrix
    # a1 = 0
    a1 = (g*mp)/((mk + mp)*(mp/(mk + mp) - 4/3))
    a2 = - g /(lp*((mp/mt)-4/3))
    
    A = np.array([[0, 1, 0,  0],
                  [0, 0, a1, 0],
                  [0, 0, 0,  1],
                  [0, 0, a2, 0]])

    # input matrix
    # b1 = 1/mt
    b1 = -(mp/(mt*(mp/mt - 4/3)) - 1)/mt
    b2 = 1 / (lp*mt*((mp/mt)-4/3))
    B = np.array([[0], [b1], [0], [b2]])
    
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Vary the value of R within the range [0.01, 0.1, 10, 100] and plot the forces 
    """
    

    R_values = [0.01, 0.1, 10, 100]
    forces_data = []

    for R in R_values:
        set_seed(args.seed) # seed for reproducibility 
    
        obs, _ = env.reset(seed=(args.seed))
        A, B = linerized_cartpole_system(mp, mk, l, g)
        K = optimal_controller(A, B, R_value=R)
        step_forces = []

        for i in range(1000):  # simulate short episode
            if time_sleep:
                time.sleep(0.01)
            action, force = apply_state_controller(K, obs)
            clipped_force = float(np.clip(force[0], -10.0, 10.0))
            abs_forces=abs(clipped_force)
            
            step_forces.append(clipped_force)

            env.unwrapped.force_mag = (abs_forces)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        forces_data.append((R, step_forces))
        print(f"Finished run for R = {R} after {len(step_forces)} steps.")


    # Plot
    matplotlib.use('TkAgg')
    plt.figure(figsize=(10, 6))
    for R, step_forces in forces_data:
        plt.plot(step_forces, label=f'R = {R}')
    plt.title("Force magnitude vs. Time for varying R values")
    plt.xlabel("Timestep(t)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("multiple_R_plot.png")
    print("Plot saved as multiple_R_plot.png")

    #TODO: 
    # ...
    return

def control(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Control using LQR
    """
    #TODO: plot the states of the system ...

    obs, _ = env.reset()    # Reset the environment for a new episode
    
    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B)    # Re-compute the optimal controller for the current R value

    for i in range(1000):

        # With Gymnasium, enable rendering via render_mode at env creation
        if time_sleep:
            time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
         # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        clipped_force = float(np.clip(force[0], -10.0, 10.0))
        abs_force = abs(clipped_force)
        
        # change magnitute of the applied force in CartPole
        env.unwrapped.force_mag = abs_force

        # apply action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if terminate and done:
            print(f'Terminated after {i+1} iterations.')
            break

# The main function
def main(args):
    # Create a Gymnasium environment
    env = gym.make(args.env, render_mode="human")

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    set_seed(args.seed)    # seed for reproducibility
    env.reset(seed=args.seed)
    
    mp, mk, l, g = env.unwrapped.masspole, env.unwrapped.masscart, env.unwrapped.length, env.unwrapped.gravity

    if args.mode == "control":
        control(env, mp, mk, l, g, args.time_sleep, terminate=True)
    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)

    env.close()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

