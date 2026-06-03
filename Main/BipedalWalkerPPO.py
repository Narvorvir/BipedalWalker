import gymnasium as gym
import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

from moduleElemantaire import PolicyNetwork, ValueNetwork, drawPlot

# =====================================================================
# PPO ALGORITHME
# =====================================================================

class AgentPPO:

    def __init__(self, env, envTest, sizeModel):

        self.env = env
        self.envTest = envTest
        self.obsDim = env.observation_space.shape[0]
        self.actionDim = env.action_space.shape[0]

        self.policyNetwork = PolicyNetwork(self.obsDim, self.actionDim,sizeModel)
        self.valueNetwork = ValueNetwork(self.obsDim, sizeModel)


    def computeGae(self,rewards, values, dones, gamma=0.99, lam=0.95):
        gae = 0
        advantages = []

        for t in reversed(range(len(rewards))):
            next_value = values[t+1] if t+1 < len(values) else 0
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    def train(self, episodes,
                gamma=0.99, lam=0.95,
                clip_eps=0.2,
                policy_lr=3e-4, value_lr=1e-3,
                steps_per_update=4000, mini_batch_size=256, epochs=10):

        # Optimize
        optPolicy = Adam(self.policyNetwork.parameters(), lr=policy_lr)
        optValue = Adam(self.valueNetwork.parameters(), lr=value_lr)

        returns_log = []
        rewards = []
        avgRewards = []

        obs, _ = self.env.reset()
        
        buffer = {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}

        print("Lancement de l'apprentissage")

        for episode in range(1,episodes):
            # --------------------------------------------------------------
            # Collecte de trajectoires jusqu’à steps_per_update
            # --------------------------------------------------------------

            episodeReward = 0
            
            for _ in range(steps_per_update):
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, log_prob = self.policyNetwork.sample_action(obs_t)
                value = self.valueNetwork(obs_t)

                next_obs, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())
                episodeReward += reward

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

            rewards.append(episodeReward)
            avgRewards.append(np.mean(rewards[-10:]))

            
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.tensor(buffer["values"], dtype=torch.float32)
            actions = torch.tensor(np.array(buffer["actions"]), dtype=torch.float32)
            old_log_probs = torch.stack(buffer["log_probs"])
            obs_tensor = torch.tensor(np.array(buffer["obs"]), dtype=torch.float32)
            dones = torch.tensor(buffer["dones"], dtype=torch.float32)

            # --------------------------------------------------------------
            # GAE : Generalized Advantage Estimation
            # --------------------------------------------------------------
            advantages = self.computeGae(buffer["rewards"], values.tolist(), dones.tolist(), gamma, lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6) # normale l'avantage

            # --------------------------------------------------------------
            # PPO Mise à jour
            # --------------------------------------------------------------
            dataset_size = len(returns)
            for _ in range(epochs):
                idx = torch.randperm(dataset_size) # eviter l'influence de l'ordre (on mélange)

                # transforme les données bruts et calcule le gradient sur des batchs (des petits morceaux)
                # accèlere le code de decouper
                for start in range(0, dataset_size, mini_batch_size):
                    end = start + mini_batch_size
                    batchIdx = idx[start:end]

                    b_obs = obs_tensor[batchIdx]
                    b_actions = actions[batchIdx]
                    b_oldLogp = old_log_probs[batchIdx]
                    b_adv = advantages[batchIdx]
                    b_returns = returns[batchIdx]

                    dist = self.policyNetwork.getDist(b_obs)

                    # Clipping
                    rawAction = torch.atanh(torch.clamp(b_actions, -0.999, 0.999))

                    # Calcule ratio
                    newLogp = dist.log_prob(rawAction).sum(dim=-1)
                    newLogp -= torch.log(1 - b_actions.pow(2) + 1e-6).sum(dim=-1)
                    ratio = torch.exp(newLogp - b_oldLogp)

                    # Clipping ratio
                    clippedRatio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                    policyLoss = -torch.min(ratio * b_adv, clippedRatio * b_adv).mean()

                    # Erreur Quadratique (MSE)
                    valuePred = self.valueNetwork(b_obs)
                    valueLoss = ((b_returns - valuePred)**2).mean()

                    optPolicy.zero_grad()
                    policyLoss.backward()
                    optPolicy.step()

                    optValue.zero_grad()
                    valueLoss.backward()
                    optValue.step()

      
            returns_log.append(np.sum(buffer["rewards"]))
            buffer = {k: [] for k in buffer}

            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes} | Return: {returns_log[-1]:.2f}")
            

        return rewards, avgRewards

    def test(self, episodes,
                stepsMax=4000):

        obs, _ = envTest.reset()
        
        for _ in range(episodes):
            for _ in range(stepsMax):
                obs_t = torch.tensor(obs, dtype=torch.float32)
                action, _ = self.policyNetwork.sample_action(obs_t)

                next_obs, _, terminated, truncated, _ = envTest.step(action.detach().numpy())
                done = terminated or truncated

                obs = next_obs
                if done:
                    obs, _ = envTest.reset()

# ===========================================
# Main
# ===========================================

if __name__ == "__main__":
    sizeModel = 64

    env = gym.make("BipedalWalker-v3")
    envTest = gym.make("BipedalWalker-v3", render_mode="human")

    # Entrainement
    agent = AgentPPO(env,envTest, sizeModel)
    rewardsHistory, avgRewards = agent.train(episodes=301)

    # Testes
    drawPlot(rewardsHistory,avgRewards, k=10)
    plt.show()

    agent.test(10)
