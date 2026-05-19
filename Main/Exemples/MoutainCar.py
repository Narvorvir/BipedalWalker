import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

pos_space = np.linspace(-1.2, 0.6, 20)
vel_space = np.linspace(-0.07, 0.07, 20)

def get_state(observartion):
    # Régler le problème d'espace continue
    pos, vel = observartion
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
    return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0,1,2]):
    # Recupère la meilleur action pour un état x
    values = np.array([Q[state, a] for a in actions])
    return np.argmax(values)

def test_Q(Q, max_episode_steps=1000,
            alpha = 0.1,  
            gamma = 0.99,
           ):
    
    env = gym.make("MountainCar-v0", max_episode_steps, render_mode="human")
    done = False
    obs, info = env.reset()
    state = get_state(obs)
    score = 0

    while not done:
        action =  max_action(Q, state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = get_state(next_obs)
        score += reward
        next_action = max_action(Q, next_state)

        # Bellman Equation
        Q[state, action] = Q[state,action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state


if __name__ == "__main__":

    max_episode_steps = 1000
    env = gym.make("MountainCar-v0", max_episode_steps)
    n_episodes = 500
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0


    # Initialise la Q Fonction
    Q = {}    
    states = []

    for pos in range(21):
        for vel in range(21):
            states.append((pos,vel))

    for state in states:
        for action in [0,1,2]:
            Q[state,action] = 0

    print(Q)

    score = 0
    total_reward = np.zeros(n_episodes)

    for i in tqdm(range(n_episodes)):
        done = False
        obs, info = env.reset()
        state = get_state(obs)
        score = 0

        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < epsilon \
                                                else max_action(Q, state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = get_state(next_obs)
            score += reward
            next_action = max_action(Q, next_state)

            Q[state, action] = Q[state,action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state

        total_reward[i] = score 
        # epsilon = max(0.01, epsilon - 2/n_episodes)
        epsilon = epsilon - 2/n_episodes if epsilon > 0.01 else 0.01

    mean_rewards = np.zeros(n_episodes)

    for t in range(n_episodes):
        mean_rewards[t] = np.mean(total_reward[max(0,t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.show()

    test_Q(Q)