import gymnasium as gym
from agent.agent import Agent
from utils.utils import plot_rewards
import time

agent = Agent()
all_rewards = []

def train_agent(num_episodes=100):
    env = gym.make("CartPole-v1", render_mode=None)  # brak renderowania
    for episode in range(num_episodes):
        total_reward = 0
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, terminated)
            agent.learn()

            obs = next_obs
        all_rewards.append(total_reward)
        agent.decent_epsilon()

    env.close()


def test_agent(num_episodes=1):
    env = gym.make("CartPole-v1", render_mode="human")  # render w oknie
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    train_agent(200)
    plot_rewards(all_rewards)