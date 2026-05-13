import gymnasium as gym
from moduleElemantaire import  dessinerCourbe
from PPO import AgentPPO
import matplotlib.pyplot as plt

# Initialisation
hardcore = False
sizeModel = 4
gamma=0.99; lam=0.95; clip_eps=0.2; policy_lr=3e-4; value_lr=1e-3

PATH = 'C:/Users/nathw/Desktop/TIPE/Code/BipedalWalkerTerminer/Models/ModelTest'
PATH_INITIAL = 'C:/Users/nathw/Desktop/TIPE/Code/BipedalWalkerTerminer/Models/ModelInitial'

env = gym.make("BipedalWalker-v3", hardcore=hardcore)
envTest = gym.make("BipedalWalker-v3", render_mode="human",  hardcore=hardcore)

agent = AgentPPO(env,envTest, sizeModel)
# agent.loadModel(PATH)
returns = agent.train(episodes=180, gamma=gamma, lam=lam,
                    clip_eps=clip_eps,
                    policy_lr=policy_lr, value_lr=value_lr)
dessinerCourbe(returns, sizeModel, gamma, lam, clip_eps, policy_lr, value_lr)
agent.save(PATH)
agent.test(10)

plt.show()
