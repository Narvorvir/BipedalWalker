import gymnasium as gym

import numpy as np

env = gym.make("BipedalWalker-v3", render_mode="human")

# VERSION ALEATOIRE

def randomPlay(iterations):
    actionSize = env.action_space.shape[0]

    for _ in range(iterations):

        _, _ = env.reset()
        done = False

        while not done:
            actions = np.random.uniform(-1,1,actionSize)
            _, _, terminated, truncated, _ = env.step(actions)

            done = terminated or truncated

randomPlay(100)