import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from moduleElemantaire import PolicyNetwork, ValueNetwork, drawPlot

# ===========================================
# A2C (Advantage Actor–Critic)
# ===========================================

def trainA2C(env : gym.Env, Actor : PolicyNetwork, Critic : ValueNetwork, 
                 episodes=5, gamma=0.99, policyLr=3e-4, valueLr=1e-3):

    actorOpt = torch.optim.Adam(Actor.parameters(), lr=policyLr)
    criticOpt = torch.optim.Adam(Critic.parameters(), lr=valueLr)

    rewardsHistory = []
    avgRewards = [] 
    advantagesEp = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False        
        episodeReward = 0
        advantages = []

        while not done:
            # 1 Estimation de l'action par l'Actor
            stateTensor = torch.FloatTensor(state)
            action, logProb = Actor.sample_action(stateTensor)

            # 2 Avancer
            nextState, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            done = terminated or truncated

            # 3 Caclule Critic
            stateValue = Critic.forward(stateTensor).squeeze()
            if not done:
                nextStateTensor = torch.FloatTensor(nextState)
                nextStateValue = Critic.forward(nextStateTensor).detach()
            else:
                nextStateValue = torch.tensor(0.)

            # 4 Avantage 
            advantage = reward + gamma * nextStateValue - stateValue

            # 5 Calcule des pertes
            # Important le detach car actor et critic doivent avoir des arbres indépendants (pour torch)
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
            state = nextState
            episodeReward += reward
            advantages.append(advantage.item()) 

        # Sauvegarde des scores pour le graphique
        if episode%30 ==0:
            advantagesEp.append(advantages)        
        rewardsHistory.append(episodeReward)
        avgReward = np.mean(rewardsHistory[-50:])
        avgRewards.append(avgReward)

        if (episode + 1) % 10 == 0:
            print(f"Épisode {episode+1}/{episodes} | Score: {episodeReward} | Moyenne (50 derniers):")

    return rewardsHistory, avgRewards, advantagesEp

def testA2C(env : gym.Env, Actor : PolicyNetwork, numEpisodes : int =3):
    for episode in range(numEpisodes):
        state, _ = env.reset()
        done = False
        totalReward = 0
        while not done:
            stateTensor = torch.FloatTensor (state)
            with torch.no_grad(): # Pas besoin de stocker les calcul pour calculer le backward (calcul du gradient)
                action, _ = Actor.sample_action(stateTensor)
            
            state, reward, terminated, truncated, _ = env.step(action.numpy())
            totalReward += reward
            done = terminated or truncated
        print(f"Test Episode {episode+1}, Total Reward: {totalReward}")
    
# ===========================================
# Main
# ===========================================

if __name__ == "__main__":
    # ENTRAINEMENT
    env = gym.make("BipedalWalker-v3", render_mode=None)
    obsDim = env.observation_space.shape[0]
    actionDim = env.action_space.shape[0]
    episodes = 1000

    env = gym.make("BipedalWalker-v3")
    Actor = PolicyNetwork(obsDim, actionDim, n=128)
    Critic = ValueNetwork(obsDim, n=128)

    rewardsHistory, avgRewards, advantages_ep = trainA2C(env, Actor, Critic, episodes, gamma=0.99, policyLr = 3e-4, valueLr=1e-3)

    # TESTE
    input("...")
    envTest = gym.make("BipedalWalker-v3", render_mode="human")
    testA2C(envTest, Actor, 5)
    envTest.close()

    plt.tight_layout()
    plt.subplot(2, 1,1)
    drawPlot(rewardsHistory, avgRewards)

    plt.subplot(2, 1, 2)
    i =0
    for advantages in advantages_ep:
        plt.plot(advantages,label=f"Avantage épisode {i*30}", alpha=0.4)
        i += 1

    plt.xlabel("Épisodes t")
    plt.ylabel("A_t")
    plt.title("Avantages suivant i ème épisode")
    plt.legend()
    plt.show()