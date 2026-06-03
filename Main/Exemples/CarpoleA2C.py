import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from moduleElemantaire import drawPlot


# ===========================================
# Actor Critic Advantage (A2C) ALGORITHME
# ===========================================

def A2C(env, actor, critic, episodes = 200, gamma=0.99, lrAct=5e-4, lrCritic=1e-3):

    rewardsHistory = []
    avgRewards = []
    advantagesEp = []
    # 1 Initialise les opt
    actorOptimize = Adam(actor.parameters(), lr=lrAct)
    criticOptimizer = Adam(critic.parameters(), lr=lrCritic)

    for episode in range(episodes):
        state, _ = env.reset()
        episodeReward = 0
        done = False
        
        advantages = []

        while not done:
            stateTensor = torch.FloatTensor(state)
            
            # 2 Calcule de l'action par l'acteur (estimation) 
            actionProbs = actor(stateTensor)
            dist = torch.distributions.Categorical(actionProbs) # créer une distribution par rapport à la proportion des valeurs et % à l'indice
            actionTensor = dist.sample()
            action = actionTensor.item() 
            
            # 3 Progretion dans l'environement 
            nextState, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 4 Calcule de la valeur de l'état actuel par le critque
            stateValue = critic(stateTensor)

            if not done:
                # Utilisation reseau du critic
                nextStateTensor = torch.FloatTensor(nextState)
                nextStateValue = critic(nextStateTensor).detach() # pas de tirage car une unique valeur de sortie
            else:
                nextStateValue = torch.tensor(0.0)
                
            # 5 Calcule de l'avantage 
            advantage = reward + gamma * nextStateValue - stateValue
            advantages.append(advantage.item())
            
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
            
            state = nextState
            episodeReward += reward

        rewardsHistory.append(episodeReward)
        avgReward = np.mean(rewardsHistory[-50:])
        avgRewards.append(avgReward)
        
        if episode%30 ==0:
            advantagesEp.append(advantages)
        
        if (episode + 1) % 10 == 0:
            print(f"Épisode {episode+1}/{episodes} | Score: {episodeReward} | Moyenne (50 derniers): {avgReward:.1f}")

    return rewardsHistory, avgRewards,advantagesEp

def testAgent(env, actor, numEpisodes=3):
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


# ===========================================
# Main
# ===========================================

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode=None)
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n

    # Réseau acteur
    actor = nn.Sequential(
        nn.Linear(stateDim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, actionDim),
        nn.Softmax(dim=-1)
    )

    # Réseau critique
    critic = nn.Sequential(
        nn.Linear(stateDim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    rewardsHistory, avgRewards,advantagesEp = A2C(env,actor, critic)
    env.close()

    # Testes
    envTest = gym.make("CartPole-v1", render_mode="human")
    testAgent(envTest, actor, numEpisodes=5)
    envTest.close()

    plt.tight_layout()

    plt.subplot(2, 1,1)
    drawPlot(rewardsHistory, avgRewards)

    plt.subplot(2, 1, 2)
    i =0
    for advantages in advantagesEp:
        plt.plot(advantages,label=f"Avantage épisode {i*30}", alpha=0.4)
        i += 1
    plt.xlabel("Épisodes t")
    plt.ylabel("A_t")
    plt.title("Avantages suivant i ème épisode")
    plt.legend()
    plt.show()