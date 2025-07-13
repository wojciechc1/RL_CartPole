import matplotlib.pyplot as plt

def plot_rewards(reward_list):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
