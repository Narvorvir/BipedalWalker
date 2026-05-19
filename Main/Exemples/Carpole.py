import numpy as np
import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Créer l'environnement
env = gym.make('CartPole-v1', render_mode=None)  # "human" seulement pour test

# Réseau acteur
actor = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='softmax')
])

# Réseau critique
critic = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Optimizers
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Hyperparamètres
num_episodes = 150
gamma = 0.99
entropy_beta = 0.01  # encourage exploration

rewards_history = []
avg_rewards = []

print("Entraînement lancé...")
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    done = False
    while not done:
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # Action selon acteur
            action_probs = actor(np.array([state]))
            action = np.random.choice(env.action_space.n, p=action_probs.numpy()[0])

            # Étape dans l'environnement
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Valeurs critiques
            state_value = critic(np.array([state]))[0, 0]
            next_state_value = critic(np.array([next_state]))[0, 0] if not done else 0.0

            # Avantage
            advantage = reward + gamma * next_state_value - state_value

            # Pertes
            actor_loss = -tf.math.log(action_probs[0, action]) * advantage
            # entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10))
            #entropy = 1 
            # actor_loss -= entropy_beta * entropy  # encourage exploration

            critic_loss = tf.square(advantage)

        # Mise à jour des poids
        actor_gradients = actor_tape.gradient(actor_loss, actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, critic.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        state = next_state
        episode_reward += reward

    # Tracking
    rewards_history.append(episode_reward)
    avg_reward = np.mean(rewards_history[-50:])  # moyenne glissante
    avg_rewards.append(avg_reward)

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}, Moyenne(50): {avg_reward:.2f}")

# ---------- Fonction pour tester le modèle ----------
def test_agent(env, actor, episodes=5, render=True):
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
            action_probs = actor(np.array([state]))
            action = np.argmax(action_probs.numpy()[0])  # exploitation pure
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Test Episode {ep+1}, Total Reward: {total_reward}")

# Test après apprentissage
print("\nTest de l'agent entraîné...")
env_test = gym.make("CartPole-v1", render_mode="human")
test_agent(env_test, actor, episodes=5, render=True)
env_test.close()

# Courbe de progression
plt.plot(rewards_history, label="Reward brut")
plt.plot(avg_rewards, label="Moyenne glissante (50)")
plt.xlabel("Épisodes")
plt.ylabel("Reward")
plt.legend()
plt.show()
