import gymnasium as gym
import time

def run_cartpole_episode():
    env = gym.make("CartPole-v1", render_mode="human")  # środowisko z renderem w oknie
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()  # losowa akcja: 0 lub 1

        print(action)

        obs, reward, terminated, truncated, info = env.step(action)
        # obs- aktualny stan,
        # reward (float),
        # terminated (bool) - Prawda, jeśli np. gra skończyła się sukcesem lub porażką,
        # truncated — przerwanie epizodu (bool) - Prawda, jeśli epizod zakończył się przedwcześnie z powodu np. limitu czasu lub innych warunków narzuconych poza logiką gry
        # info — dodatkowe informacje (dict)

        done = terminated or truncated

        time.sleep(0.02)  # żeby animacja nie latala za szybko

    env.close()

if __name__ == "__main__":
    for i in range(10):
        run_cartpole_episode()
        print(i)