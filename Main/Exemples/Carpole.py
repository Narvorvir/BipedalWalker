import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

# Créer l'environnement pour récupérer les dimensions
env = gym.make('CartPole-v1', render_mode=None)
state_dim = env.observation_space.shape[0]  # Dynamique (ici égal à 4)
action_dim = env.action_space.n             # Dynamique (ici égal à 2)

# Réseau acteur (Modifications : dimensions d'entrée et retrait des crochets [])
actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, action_dim),
    nn.Softmax(dim=-1) # Spécifier la dimension pour le Softmax est fortement recommandé
)

# Réseau critique (Modifications : dimensions d'entrée, retrait des crochets et de la dernière ReLU)
# Note : Pas de ReLU en sortie du critique pour lui permettre de prédire des valeurs négatives si besoin.
critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Optimizers (Modifications : .parameters() ajouté et 'learning_rate' changé en 'lr')
# actor_optimizer = Adam(actor.parameters(), lr=0.001)
# critic_optimizer = Adam(critic.parameters(), lr=0.002)
actorOptimize = Adam(actor.parameters(), lr=0.0005)
criticOptimizer = Adam(critic.parameters(), lr=0.001)

# Hyperparamètres
NUM_EPISODES = 300 
GAMMA = 0.99 # facteur d'atténuation à quel point les anciens états sont importants

rewardsHistory = []
avgRewards = []

print("Entraînement lancé...")
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        stateTensor = torch.FloatTensor(state)
        
        # 2 Calcule de l'action par l'acteur (estimation) 
        actionProbs = actor(stateTensor)
        dist = torch.distributions.Categorical(actionProbs) # créer une distribution par rapport à la proportion des valeurs et % à l'indice
        actionTensor = dist.sample()
        action = actionTensor.item() 
        
        # 3 Progretion dans l'environement 
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 4 Calcule de la valeur de l'état actuel par le critque
        stateValue = critic(stateTensor)

        if not done:
            # Utilisation reseau du critic
            nextStateTensor = torch.FloatTensor(next_state)
            nextStateValue = critic(nextStateTensor).detach() # pas de tirage car une unique valeur de sortie
        else:
            nextStateValue = torch.tensor(0.0) # reset
            
        # 5 Calcule de l'avantage 
        advantage = reward + GAMMA * nextStateValue - stateValue
        
        # 6 Calcul des Pertes (Losses)
        log_prob = dist.log_prob(actionTensor)
        actor_loss = -log_prob * advantage.detach() # (- car ln) et contient à la fois le tensor et le graphe de calcul
        critic_loss = advantage.pow(2)
        
        # 7 Mise à jour des poids des réseaux de neurones Actor et Critic
        # met à zero les gradients
        actorOptimize.zero_grad()  
        criticOptimizer.zero_grad()
        
        # calcule gradient
        actor_loss.backward()
        critic_loss.backward()
        
        # met à jour les poids des reseaux de neurones
        actorOptimize.step()
        criticOptimizer.step()
        
        # Avancer d'un pas
        state = next_state
        episode_reward += reward

    # Sauvegarde des scores pour le graphique (Ajouté)
    rewardsHistory.append(episode_reward)
    avgReward = np.mean(rewardsHistory[-50:])
    avgRewards.append(avgReward)
    
    if (episode + 1) % 10 == 0:
        print(f"Épisode {episode+1}/{NUM_EPISODES} | Score: {episode_reward} | Moyenne (50 derniers): {avgReward:.1f}")

env.close()

# ---------- Fonction pour tester le modèle ----------
def test_agent(env, actor, numEpisodes=3):
    for episode in range(numEpisodes):
        state, _ = env.reset()
        done = False
        totalReward = 0
        while not done:
            stateTensor = torch.FloatTensor(state)
            with torch.no_grad(): # Pas besoin de stocker les calcul pour calculer le backward (calcul du gradient)
                actionProbs = actor(stateTensor)
            
            action = torch.argmax(actionProbs).item()  
            state, reward, terminated, truncated, _ = env.step(action)
            totalReward += reward
            done = terminated or truncated
        print(f"Test Episode {episode+1}, Total Reward: {totalReward}")

# Test après apprentissage
print("Test de l'agent entraîné (Fermer la fenêtre du jeu pour voir le graphique)...")
env_test = gym.make("CartPole-v1", render_mode="human")
test_agent(env_test, actor, numEpisodes=5)
env_test.close()

# Courbe de progression
plt.figure(figsize=(10, 5))
plt.plot(rewardsHistory, label="Reward brut", alpha=0.4)
plt.plot(avgRewards, label="Moyenne glissante (50)", color="red", linewidth=2)
plt.xlabel("Épisodes")
plt.ylabel("Reward")
plt.title("Progression de l'entraînement de l'agent")
plt.legend()
plt.show()