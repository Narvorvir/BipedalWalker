import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from moduleElemantaire import PolicyNetwork, ValueNetwork

NUM_EPISODE = 5


# ===========================================
# A2C (Advantage Actor–Critic)
# ===========================================

def trainA2C(env : gym.Env, Actor : PolicyNetwork, Critic : ValueNetwork, 
                 episodes=5, gamma=0.99, policyLr=3e-4, valueLr=1e-3, lamb=0.95, actionNumber= 1000):

    actorOpt = torch.optim.Adam(Actor.parameters(), lr=policyLr)
    criticOpt = torch.optim.Adam(Critic.parameters(), lr=valueLr)

    rewardsHistory = []
    avgRewards = [] 
    advtangeAvgs = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False        
        episodeReward = 0
        advantageAvg = 0
        advantage = torch.tensor(0)

        buffer = {}
        buffer["rewards"] = []
        i = 0 

        for i in range(1,actionNumber):
            if done: break

            # 1. Estimation de l'action par l'Actor (foward, distribution multinomiale, tirage aléatoire)
            stateTensor = torch.tensor(state)
            action, logProb = Actor.sample_action(stateTensor)

            # 2. Avancer
            nextState, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            done = terminated or truncated

            # 3. Caclule Critic
            stateValue = Critic.forward(stateTensor)
            if not done:
                nextStateTensor = torch.FloatTensor(nextState)
                nextStateValue = Critic.forward(nextStateTensor).detach()
            else:
                nextStateValue = torch.tensor(0)

            # 1. Prédiction Actor/Critic
            


            if i % 5 == 0 :
                
                # 4. Avantage 
                # advantage = reward + gamma * (nextStateValue - stateValue)
                td = reward + gamma * nextStateValue - stateValue 
                advantage = td + gamma * lamb * advantage.detach()
                # 5. Calcule des pertes
                # Important le detach car actor et critic doivent avoir des arbres indépendants
                actorLoss = - logProb * advantage.detach() 
                criticLoss = advantage.pow(2)


                # 6. Mettre à jour les réseaux
                actorOpt.zero_grad()
                criticOpt.zero_grad()

                actorLoss.backward()
                criticLoss.backward()

                actorOpt.step()
                criticOpt.step()

                # 7. Avancer 
                # Avancer d'un pas
                state = nextState
                episodeReward += reward
                advantageAvg += advantage.item()

        # Sauvegarde des scores pour le graphique (Ajouté)
        advtangeAvgs.append(advantageAvg)
        rewardsHistory.append(episodeReward)
        avgReward = np.mean(rewardsHistory[-50:])
        avgRewards.append(avgReward)

        if (episode + 1) % 10 == 0:
            print(f"Épisode {episode+1}/{episodes} | Score: {episodeReward} | Moyenne (50 derniers): {avgReward:.1f} Avantage {advantageAvg}")

    return rewardsHistory, avgRewards, advtangeAvgs

def testA2C(env : gym.Env, Actor : PolicyNetwork, numEpisodes : int =3):
    for episode in range(numEpisodes):
        state, _ = env.reset()
        done = False
        totalReward = 0
        while not done:
            stateTensor = torch.FloatTensor (state)
            with torch.no_grad(): # Pas besoin de stocker les calcul pour calculer le backward (calcul du gradient)
                action, _ = Actor.sample_action(stateTensor)
            
            state, reward, terminated, truncated, _ = env.step(action)
            totalReward += reward
            done = terminated or truncated
        print(f"Test Episode {episode+1}, Total Reward: {totalReward}")
    
# ===========================================
# Main
# ===========================================

# ENTRAINEMENT
env = gym.make("BipedalWalker-v3", render_mode=None)
obsDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]

episodes = 1000

env = gym.make("BipedalWalker-v3")
Actor = PolicyNetwork(obsDim, actionDim, n=128)
Critic = ValueNetwork(obsDim, n=128)

rewardsHistory, avgRewards, advtangeAvgs = trainA2C(env, Actor, Critic, episodes, gamma=0.99, policyLr = 0.001, valueLr=0.001)

# TESTE
print("Test de l'agent entraîné (Fermer la fenêtre du jeu pour voir le graphique)...")
input("...")
envTest = gym.make("BipedalWalker-v3", render_mode="human")
testA2C(envTest, Actor, 2)
envTest.close()


# Courbe de progression
plt.figure(figsize=(10, 5))
# print(advtangeAvgs)
plt.plot(rewardsHistory, label="Reward brut", alpha=0.4)
plt.plot(avgRewards, label="Moyenne glissante (50)", color="red", linewidth=2)
plt.plot(advtangeAvgs, label="Avantage Moyenne", color="green", linewidth=2)
plt.xlabel("Épisodes")
plt.ylabel("Reward")
plt.title("Progression de l'entraînement de l'agent")
plt.legend()
plt.show()