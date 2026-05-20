import gymnasium as gym
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

NOMBRE_EPISODE = 5

# ===========================================
# Environnement
# ===========================================

# ===========================================
# Politique gaussienne
# ===========================================

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        std = torch.exp(self.log_std)
        return mu, std

    def get_dist(self, obs):
        mu, std = self.forward(obs)
        return torch.distributions.Normal(mu, std)

    def sampleAction(self, obs):
        dist = self.get_dist(obs)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob

# ===========================================
# Critique = V(s)
# ===========================================
class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

# ===========================================
# A2C (Advantage Actor–Critic)
# ===========================================

def actor_critic(env : gym.Env, ActorPolicy : GaussianPolicy, ValueNet : ValueNet, episodes, gamma=0.99, lr_policy=3e-4, lr_value=1e-3):

    optim_policy = Adam(ActorPolicy.parameters(), lr=lr_policy)
    optim_value = Adam(ValueNet.parameters(), lr=lr_value)

    returns_list = []

    for ep in tqdm(range(episodes)):
        obs, info = env.reset()
        done = False
        ep_return = 0

        totalEpisodeReward = 0

        while not done:

            obs_t = torch.tensor(obs, dtype=torch.float32)

            value = ValueNet(obs_t)
            action, log_prob = ActorPolicy.sampleAction(obs_t)

            next_obs, reward, terminated, truncated, info = env.step(action.detach().numpy())
            done = terminated or truncated

            with torch.no_grad():
                next_value = ValueNet(torch.tensor(next_obs, dtype=torch.float32)) if not done else 0.0

            td_target = reward + gamma * next_value
            td_error  = td_target - value

            adv = td_error.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            policy_loss = -(log_prob * adv).mean()
            
            # policy_loss = -(log_prob * adv + 0.01 * entropy).mean()

            optim_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(ActorPolicy.parameters(), 0.5)
            optim_policy.step()

            value_loss = td_error.pow(2).mean()
            optim_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(ValueNet.parameters(), 0.5)
            optim_value.step()

            # <-- tu l'avais oublié
            ep_return += reward

            obs = next_obs


        # fin d’épisode
        totalEpisodeReward += ep_return
        returns_list.append(ep_return)


        totalEpisodeReward += reward

        returns_list.append(ep_return)

        if ep % NOMBRE_EPISODE == 0:
            print(f"Episode {ep} | Return: {totalEpisodeReward/NOMBRE_EPISODE:.6f}")
            totalEpisodeReward = 0            

    return returns_list


# ===========================================
# Main
# ===========================================

env = gym.make("BipedalWalker-v3", render_mode=None)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


actorPolicy = GaussianPolicy(obs_dim, action_dim)
criticPolicy = ValueNet(obs_dim)

episodes = 200

returns = actor_critic(env, actorPolicy, criticPolicy, episodes)

env = gym.make("BipedalWalker-v3", render_mode="human")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
returns = actor_critic(env, actorPolicy, criticPolicy, 10)

plt.plot(returns)
plt.xlabel("Épisode")
plt.ylabel("Retour")
plt.title("Actor–Critic (A2C) sur BipedalWalker-v3")
plt.show()
