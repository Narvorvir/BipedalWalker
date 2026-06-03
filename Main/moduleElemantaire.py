import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===============================================================
# PolicyNetwork et ValueNetwork et d'autres fonctions utiles
# ===============================================================

class PolicyNetwork(nn.Module):
    def __init__(self, obsDim, actionDim, n=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obsDim, 4*n),
            nn.ReLU(),
            nn.Linear(4*n, 2*n),
            nn.ReLU(),
        )
        self.meanLayer = nn.Linear(2*n, actionDim)
        self.logStd = nn.Parameter(torch.zeros(actionDim))

    def forward(self, obs : torch.Tensor) -> torch.Tensor:
        x = self.network(obs)
        mean = self.meanLayer(x)
        std = torch.exp(self.logStd)
        return mean, std

    def getDist(self, obs):
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)

    def sample_action(self, obs):
        distribution = self.getDist(obs)
        rawAction = distribution.sample() 
        action = torch.tanh(rawAction) # peut aller de +inf à -inf donc on le remet entr 1 et -1

        logProb = distribution.log_prob(rawAction).sum(dim=-1)
        logProb -= torch.log(torch.clamp(1 - action.pow(2), min=1e-6)).sum(dim=-1)

        return action, logProb


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, n=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 4*n),
            nn.ReLU(),
            nn.Linear(4*n, 2*n),
            nn.ReLU(),
            nn.Linear(2*n, 1)
        )

    def forward(self, obs : torch.tensor) -> torch.tensor:
        return self.network(obs).squeeze(-1)


def drawPlot(rewardsHistory, avgRewards, k=50):
    plt.plot(rewardsHistory, label="Reward brut", alpha=0.4)
    plt.plot(avgRewards, label=f"Moyenne glissante ({k})", color="red", linewidth=2)
    plt.xlabel("Épisodes")
    plt.ylabel("Reward")
    plt.title("Progression de la récompense")
    plt.legend()