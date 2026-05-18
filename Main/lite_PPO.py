import gymnasium as gym
import torch
from torch.optim import Adam
import numpy as np

from moduleElemantaire import PolicyNetwork, ValueNetwork, dessinerCourbe, mean

import time

# =====================================================================
# PPO ALGORITHM
# =====================================================================

class AgentPPO:

    def __init__(self, env, envTest, sizeModel):

        self.env = env
        self.envTest = envTest
        self.obsDim = env.observation_space.shape[0]
        self.actionDim = env.action_space.shape[0]

        self.policyNetwork = PolicyNetwork(self.obsDim, self.actionDim,sizeModel)
        self.valueNetwork = ValueNetwork(self.obsDim, sizeModel)



    def train(self, episodes,
                gamma=0.99, lam=0.95,
                clip_eps=0.2,
                policy_lr=3e-4, value_lr=1e-3,
                steps_per_update=4000, mini_batch_size=256, epochs=10):

        # Optimize
        opt_policy = Adam(self.policyNetwork.parameters(), lr=policy_lr)
        opt_value = Adam(self.valueNetwork.parameters(), lr=value_lr)

        returns_log = []

        obs, _ = self.env.reset()
        
        
        buffer = {
            "obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []
        }

        print("Lancement de l'apprentissage")

        for episode in range(1,episodes):

            # --------------------------------------------------------------
            # Collecte de trajectoires jusqu’à steps_per_update
            # --------------------------------------------------------------
            
            for _ in range(steps_per_update):
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, log_prob = self.policyNetwork.sample_action(obs_t)
                value = self.valueNetwork(obs_t)

                next_obs, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())
                done = terminated or truncated

                buffer["obs"].append(obs)
                buffer["actions"].append(action.detach().numpy())
                buffer["log_probs"].append(log_prob.detach())
                buffer["rewards"].append(reward)
                buffer["dones"].append(done)
                buffer["values"].append(value.item())
                
                obs = next_obs
                if done:
                    obs, _ = self.env.reset()

            returns = []
            G = 0
            for r, d in zip(reversed(buffer["rewards"]), reversed(buffer["dones"])):
                if d : G = 0
                G = r + gamma * G # Calcule TD
                returns.insert(0, G)

            
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.tensor(buffer["values"], dtype=torch.float32)
            actions = torch.tensor(np.array(buffer["actions"]), dtype=torch.float32)
            old_log_probs = torch.stack(buffer["log_probs"])
            obs_tensor = torch.tensor(np.array(buffer["obs"]), dtype=torch.float32)
            dones = torch.tensor(buffer["dones"], dtype=torch.float32)

            # --------------------------------------------------------------
            # GAE : Generalized Advantage Estimation
            # --------------------------------------------------------------
            advantages = self.compute_gae(buffer["rewards"], values.tolist(), dones.tolist(), gamma, lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6) # normale l'avantage

            # --------------------------------------------------------------
            # PPO updates
            # --------------------------------------------------------------
            dataset_size = len(returns)
            for _ in range(epochs):
                idx = torch.randperm(dataset_size) # eviter l'influence de l'ordre (on mélange)

                # transforme les données bruts et calcule le gradient sur des batchs (des petits morceaux)
                # accèlere le code de decouper
                for start in range(0, dataset_size, mini_batch_size):
                    end = start + mini_batch_size
                    batch_idx = idx[start:end]

                    b_obs = obs_tensor[batch_idx]
                    b_actions = actions[batch_idx]
                    b_old_logp = old_log_probs[batch_idx]
                    b_adv = advantages[batch_idx]
                    b_returns = returns[batch_idx]

                    dist = self.policyNetwork.get_dist(b_obs)

                    # Clipping
                    raw_action = torch.atanh(torch.clamp(b_actions, -0.999, 0.999))

                    # Calcule ratio
                    new_logp = dist.log_prob(raw_action).sum(dim=-1)
                    new_logp -= torch.log(1 - b_actions.pow(2) + 1e-6).sum(dim=-1)
                    ratio = torch.exp(new_logp - b_old_logp)

                    # Clipping ratio
                    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                    policy_loss = -torch.min(ratio * b_adv, clipped_ratio * b_adv).mean()

                    # Erreur Quadratique (MSE)
                    value_pred = self.valueNetwork(b_obs)
                    value_loss = ((b_returns - value_pred)**2).mean()

                    opt_policy.zero_grad()
                    policy_loss.backward()
                    opt_policy.step()

                    opt_value.zero_grad()
                    value_loss.backward()
                    opt_value.step()

            # --------------------------------------------------------------
            # Logging
            # --------------------------------------------------------------
            returns_log.append(np.sum(buffer["rewards"]))
            buffer = {k: [] for k in buffer}

            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes} | Return: {returns_log[-1]:.2f}")
            

        return returns_log



# Initialisation
hardcore = False
sizeModel = 64
gamma=0.99; lam=0.95; clip_eps=0.2; policy_lr=3e-4; value_lr=1e-3
# gamma=0.99; lam=0.95; clip_eps=0.2; policy_lr=1e-4; value_lr=5e-4

PATH = 'C:/Users/nathw/Desktop/TIPE/Code/BipedalWalkerTerminer/Models/ModelTest'
PATH_INITIAL = 'C:/Users/nathw/Desktop/TIPE/Code/BipedalWalkerTerminer/Models/ModelInitial'

env = gym.make("BipedalWalker-v3", hardcore=hardcore)
envTest = gym.make("BipedalWalker-v3", render_mode="human",  hardcore=hardcore)

agent = AgentPPO(env,envTest, sizeModel)
# agent.loadModel(PATH)
returns = agent.train(episodes=100, gamma=gamma, lam=lam,
                    clip_eps=clip_eps,
                    policy_lr=policy_lr, value_lr=value_lr)
# dessinerCourbe(returns, sizeModel, gamma, lam, clip_eps, policy_lr, value_lr)
# agent.save(PATH)
agent.test(10)