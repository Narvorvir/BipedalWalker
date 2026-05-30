import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from moduleElemantaire import ValueNetwork, drawPlot

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, n=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 4*n),
            nn.ReLU(),
            nn.Linear(4*n, 2*n),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(2*n, action_dim)
        self.log_std_layer = nn.Linear(2*n, action_dim)

    def forward(self, obs):
        x = self.network(obs)
        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean, std

    def get_dist(self, obs):
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)

    def sample_action(self, obs):
        dist = self.get_dist(obs)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-6)).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

def trainA2C(env, actor, critic, episodes=100, gamma=0.99,
             policy_lr=3e-4, value_lr=1e-3, entropy_coef=0.001):

    actor_opt = torch.optim.Adam(actor.parameters(), lr=policy_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=value_lr)

    rewards_history = []

    for episode in range(episodes):
        state, _ = env.reset()

        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        entropies = []

        done = False
        total_reward = 0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32)

            action, log_prob, entropy = actor.sample_action(state_t)
            value = critic(state_t)

            next_state, reward, terminated, truncated, _ = env.step(
                action.detach().cpu().numpy()
            )
            done = terminated or truncated

            states.append(state_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            entropies.append(entropy)

            state = next_state
            total_reward += reward

        # =========================
        # Calcul des returns
        # =========================
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)

        # Advantage
        advantages = returns - values.detach()

        # Normalisation (CRUCIAL)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # =========================
        # Loss
        # =========================
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = (returns - values).pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss

        # =========================
        # Update
        # =========================
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        loss.backward()

        # gradient clipping (TRÈS important)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)

        actor_opt.step()
        critic_opt.step()

        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episode {episode+1} | Reward: {total_reward:.1f} | Avg50: {avg:.1f}")

    return rewards_history

# ENTRAINEMENT
env = gym.make("BipedalWalker-v3", render_mode=None)
obsDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]

episodes = 100

env = gym.make("BipedalWalker-v3")
Actor = PolicyNetwork(obsDim, actionDim, n=128)
Critic = ValueNetwork(obsDim, n=128)

rewardsHistory, avgRewards, advtangeAvgs = trainA2C(env, Actor, Critic, episodes)

# TESTE
# print("Test de l'agent entraîné (Fermer la fenêtre du jeu pour voir le graphique)...")
# input("...")
# envTest = gym.make("BipedalWalker-v3", render_mode="human")
# testA2C(envTest, Actor, 5)
# envTest.close()

drawPlot(rewardsHistory, avgRewards)