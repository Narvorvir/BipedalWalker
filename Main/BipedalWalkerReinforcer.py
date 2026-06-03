import gymnasium as gym
import torch
from torch.optim import Adam
import numpy as np
from moduleElemantaire import drawPlot, PolicyNetwork

import matplotlib.pyplot as plt

# ===========================================
# Reinforcer ALGORITHME
# ===========================================

def reinforce(env, policy, episodes, gamma=0.99, lr=3e-4): # (les valeurs sont classiques en RL)
    optimizer = Adam(policy.parameters(), lr=lr)
    rewardHistory = []
    avgRewards = []
    GainEp = []

    for episode in range(episodes):
        obs, _ = env.reset()
        epRewards = []
        epLogProbs = []
        done = False

        # On fait un épisode en entier
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            action, log_prob = policy.sample_action(obs_t)
            action_np = action.detach().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            epRewards.append(reward)
            epLogProbs.append(log_prob)
            obs = next_obs

        epReturn = sum(epRewards)
        rewardHistory.append(epReturn)
        avgReward = np.mean(rewardHistory[-50:])
        avgRewards.append(avgReward)

        # Calcul du Gain
        returns = []
        G = 0
        for r in reversed(epRewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()

        GainEp.append(G)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            # Normalisation des returns avec Z-score
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Calcule de la perte
        logProb_t = torch.stack(epLogProbs)
        loss = -(logProb_t * returns_t).mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode % 10 == 0):
            print(f"Episode {episode}/{episodes} | Total Reward: {epReturn:.2f} | Average (50ep): {avgReward:.2f}")
        
    return rewardHistory, avgRewards, GainEp


# ===========================================
# Main
# ===========================================

if __name__ == "__main__":

    env = gym.make("BipedalWalker-v3")
    obsDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    policy = PolicyNetwork(obsDim, actionDim)

    episodes = 500
    rewardHistory, avgRewards, G_ep = reinforce(env, policy, episodes)

    # Dessier plot
    plt.subplot(2, 1,1)
    drawPlot(rewardHistory, avgRewards)

    plt.subplot(2, 1,2)
    plt.plot(G_ep, label="Gain brut", alpha=0.4)
    plt.xlabel("Épisodes t")
    plt.ylabel("G_t")
    plt.title("Gain brut")
    plt.legend()
    plt.show()