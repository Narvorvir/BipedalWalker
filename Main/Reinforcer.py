import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from moduleElemantaire import drawPlot

# CONCLUSIONS
# Ne fonctionne pas

rewards_list = []

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Cas simplifier on garde le même écart type pour tous les états
        # On s'intéresse qu'à prédirer aux moyennes afin de tracer les courbes de gauss 

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.moyennes = nn.Linear(64, action_dim)

        self.log_ecartType = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        x = self.network(obs)
        moyennes = self.moyennes(x) # retourne les moyennes
        ecartTypes = torch.exp(self.log_ecartType)

        return moyennes, ecartTypes

    def sample_action(self, obs):
        moyennes, ecartTypes = self.forward(obs)
        distributions = torch.distributions.Normal(moyennes, ecartTypes)
        action = distributions.sample() # tire au hasard une valeur de chaque Gaussienne
        log_prob = distributions.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, -1, 1)  # BipedalWalker action range

        return action, log_prob # utile pour calculer le gradient


def reinforce(env, policy, episodes, gamma=0.99, lr=3e-5):
    # lr=3e-4 
    # lr=3e-3 (semble plus stable)

    optimizer = Adam(policy.parameters(), lr=lr)
    returnList = []
    avgRewards = []
    rewardHistory = []

    for episode in range(episodes):
        obs, _ = env.reset()
        epRewards = []
        epLogProbs = []
        done = False

        # 1. Intéragit avec l'environement
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)

            action, log_prob = policy.sample_action(obs_t)
            action_np = action.detach().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            epRewards.append(reward)
            epLogProbs.append(log_prob)
            obs = next_obs

        rewardHistory.append(sum(epRewards))
        returns = []
        avgReward = np.mean(rewardHistory[-50:])
        avgRewards.append(avgReward)

        G = 0 # return
        for r in reversed(epRewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()

        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)  # normalize

        # Calcule de la perte
        log_probs_t = torch.stack(epLogProbs)
        loss = -(log_probs_t * returns_t).sum()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        returnList.append(sum(rewardHistory))

        if (episode % 10 == 0):
            print(f"Episode {episode}/{episodes} | Return {returnList[-1]}")
        
    return rewardHistory, avgRewards


# Initialisation
env = gym.make("BipedalWalker-v3", render_mode=None)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy = GaussianPolicy(obs_dim, action_dim)

episodes = 500
rewardHistory, avgRewards = reinforce(env, policy, episodes)

drawPlot(rewardHistory, avgRewards)