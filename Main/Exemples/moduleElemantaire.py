import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import os


# =====================================================================
# Réseaux : Policy & Value function
# =====================================================================


class PolicyNetwork(nn.Module):
    """
    Réseau de policy pour travailler sur un environement continu.
    On utilise le fait qu'une gausienne est seulement définit par son écart type (std) et son espérance (mean)
    L'objectif est de construire une gaussienne pour chaque action c'est à dire générer un écart type et une
    espérance à partir d'un réseau neuronne.

    Paramètres :
        obs_dim (int) : dimension de l'état (observation)
        action_dim (int) : dimension de l'action (output)
    """
    def __init__(self, obs_dim, action_dim, n=64):
        super().__init__()
        # self.network = nn.Sequential(
        #     nn.Linear(obs_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        # )
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 4*n),
            nn.ReLU(),
            nn.Linear(4*n, 2*n),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(2*n, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs : torch.Tensor) -> torch.Tensor:
        """
        Calcule la moyenne et l'écart-type pour un état donné.

        Args:
            obs (torch.Tensor) : état d'entrée

        Returns:
            mean (torch.Tensor) : moyenne de la distribution gaussienne
            std (torch.Tensor) : écart-type (std) de la distribution
        """
        x = self.network(obs)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std

    def getDist(self, obs):
        """
        Retourne la distribution gaussienne des actions pour un état donné.

        """
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)

    def sample_action(self, obs):
        """
        Échantillonne une action selon la policy et calcule sa log-probabilité.
        """
        distribution = self.getDist(obs)
        # rawAction = distribution.rsample() 
        rawAction = distribution.sample() 
        action = torch.tanh(rawAction) # peut aller de +inf à -inf donc on le remet entr 1 et -1

        # correction du log_prob pour le tanh
        log_prob = distribution.log_prob(rawAction).sum(dim=-1)
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-6)).sum(dim=-1)

        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Réseau de critic (Value function)

    Paramètres :
        obs_dim (int) : dimension de l'état (observation)
    """
    def __init__(self, obs_dim, n=64):
        super().__init__()
        # self.network = nn.Sequential(
        #     nn.Linear(obs_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 4*n),
            nn.ReLU(),
            nn.Linear(4*n, 2*n),
            nn.ReLU(),
            nn.Linear(2*n, 1)
        )

    def forward(self, obs : torch.tensor) -> torch.tensor:
        """
        Estime la valeur d'un état.

        Args:
            obs (torch.Tensor) : état d'entrée

        Returns:
            value (torch.Tensor) : valeur estimée V(s)
        """
        return self.network(obs).squeeze(-1)

def mean(t):
    return sum(t) / len(t)


def saveTab(path, t):
    """Sauvegarde un tableau numpy ou une liste au format .npy."""
    try:
        # Créer le dossier parent si il n'existe pas
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Conversion en array numpy si c'est une liste
        array_to_save = np.array(t)
        np.save(path, array_to_save)
        print(f"Tableau sauvegardé avec succès dans : {path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")

def readTab(path):
    """Lit un fichier .npy et retourne le tableau."""
    try:
        if os.path.exists(path):
            t = np.load(path)
            print(f"Tableau chargé depuis : {path} (Taille: {len(t)})")
            return t.tolist()
        else:
            print(f"Le fichier {path} n'existe pas.")
            return []
    except Exception as e:
        print(f"Erreur lors de la lecture : {e}")
        return []


def drawPlot(rewardsHistory, avgRewards):
    # Courbe de progression
    # plt.figure(figsize=(10, 5))
    plt.plot(rewardsHistory, label="Reward brut", alpha=0.4)
    plt.plot(avgRewards, label="Moyenne glissante (50)", color="red", linewidth=2)
    plt.xlabel("Épisodes")
    plt.ylabel("Reward")
    plt.title("Progression de la récompense")
    plt.legend()
