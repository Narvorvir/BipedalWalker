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

    def forward(self, obs):
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

    def get_dist(self, obs):
        """
        Retourne la distribution gaussienne des actions pour un état donné.

        """
        mean, std = self.forward(obs)
        return torch.distributions.Normal(mean, std)

    def sample_action(self, obs):
        """
        Échantillonne une action selon la policy et calcule sa log-probabilité.
        """
        distribution = self.get_dist(obs)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)

        # correction du log_prob pour le tanh
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

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

    def forward(self, obs):
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

def dessinerCourbe(returns,sizeModel, gamma, lam, clip_eps, policy_lr, value_lr):
    params_text = (f"Hyperparamètres:\n"
                   f"Size: {sizeModel} | γ: {gamma} | λ: {lam}\n"
                   f"Clip: {clip_eps} | LR_pol: {policy_lr} | LR_val: {value_lr}")
    
    plt.plot(returns, "ob")
    plt.xlabel("Episode")
    plt.ylabel("Return estimé (somme rewards collectées)")
    plt.title("Progression de l'apprentissage PPO (BipedalWalker)\n"+params_text, fontsize=15, pad=20)
    plt.plot(returns, "ob")
    
    # plt.figtext(0.6, 0.1, params_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout() # ajuste les marges

def init_plot(title="Reward par épisode", xlabel="Episode", ylabel="Reward"):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '+')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show(block=False)  # Affiche la fenêtre mais ne bloque pas
    return fig, ax, line

def update_plot(line, rewards_list):
    line.set_xdata(range(len(rewards_list)))
    line.set_ydata(rewards_list)
    line.axes.relim()
    line.axes.autoscale_view()
    line.figure.canvas.draw()
    line.figure.canvas.flush_events()  # Rafraîchit sans forcer le focus

