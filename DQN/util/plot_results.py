import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set(style="darkgrid")

reward_file = "best_model/ep_reward.json"

with open(reward_file) as f:
    ep_rewards = json.load(f)

def moving_average(x, w=8):
    return np.convolve(x, np.ones(w), 'valid') / w

plt.plot(moving_average(ep_rewards))
plt.show()