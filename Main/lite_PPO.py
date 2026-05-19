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
actor_optimizer = Adam(actor.parameters(), lr=0.0005)
critic_optimizer = Adam(critic.parameters(), lr=0.001)
# Hyperparamètres
num_episodes = 300
gamma = 0.99

rewards_history = []
avg_rewards = []

print("Entraînement lancé...")
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # 1 Préparation de l'état en tenseur PyTorch
        # state_tensor = torch.FloatTensor(state).unsqueeze(0) 
        state_tensor = torch.FloatTensor(state)
        # print(state_tensor)
        
        # 2 Action selon l'acteur : on met dans le réseau de neurones 
        actionProbs = actor(state_tensor)
        dist = torch.distributions.Categorical(actionProbs) # créer une distribution par rapport à la proportion des valeurs et % à l'indice
        actionTensor = dist.sample()
        action = actionTensor.item() 
        
        # 3 Étape dans l'environnement
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 4 Valeurs critiques
        # state_value = critic(state_tensor).squeeze() 
        stateValue = critic(state_tensor)

        if not done:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_state_value = critic(next_state_tensor).squeeze().detach()
        else:
            next_state_value = torch.tensor(0.0)
            
        # 5. Avantage 
        advantage = reward + gamma * next_state_value - stateValue
        
        # 6. Calcul des Pertes (Losses)
        log_prob = dist.log_prob(actionTensor)
        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        
        # 7. Mise à jour des poids 
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()
        
        actor_optimizer.step()
        critic_optimizer.step()
        
        # Avancer d'un pas
        state = next_state
        episode_reward += reward

    # Sauvegarde des scores pour le graphique (Ajouté)
    rewards_history.append(episode_reward)
    avg_reward = np.mean(rewards_history[-50:])
    avg_rewards.append(avg_reward)
    
    if (episode + 1) % 10 == 0:
        print(f"Épisode {episode+1}/{num_episodes} | Score: {episode_reward} | Moyenne (50 derniers): {avg_reward:.1f}")

env.close()

# ---------- Fonction pour tester le modèle ----------
def test_agent(env, actor, episodes=3):
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad(): # Pas besoin de stocker les gradients pendant le test
                action_probs = actor(state_tensor)
            
            action = torch.argmax(action_probs).item()  
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Test Episode {ep+1}, Total Reward: {total_reward}")

# Test après apprentissage
print("\nTest de l'agent entraîné (Fermer la fenêtre du jeu pour voir le graphique)...")
env_test = gym.make("CartPole-v1", render_mode="human")
test_agent(env_test, actor, episodes=5)
env_test.close()

# Courbe de progression
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label="Reward brut", alpha=0.4)
plt.plot(avg_rewards, label="Moyenne glissante (50)", color="red", linewidth=2)
plt.xlabel("Épisodes")
plt.ylabel("Reward")
plt.title("Progression de l'entraînement de l'agent")
plt.legend()
plt.show()