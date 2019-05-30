
# coding: utf-8

# In[ ]:


import numpy as np
from collections import deque


# In[ ]:


class RandomSearchAgent:
    def __init__(self, agent_model, episodes=500, max_t=1000, gamma=1.0, print_every=10, sigma=0.5):
        self.episodes = episodes
        self.max_t = max_t
        self.gamma = gamma
        self.print_every = print_every
        self.scores = []
        self.sigma = sigma
        self.agent_model = agent_model
        self.scores_deque = deque(maxlen=100)
    def train_model(self):
        best_rewards = -np.inf
        for episode in range(1, self.episodes+1):
            weights = np.random.randn(self.agent_model.get_weights_dim())
            rewards = np.array(self.agent_model.evaluate(weights, self.gamma, self.max_t))
            if rewards > best_rewards:
                best_rewards = rewards
                self.agent_model.set_weights(weights)
                self.best_weight = weights
            reward = self.agent_model.evaluate(self.best_weight, self.gamma)
            self.scores_deque.append(reward)
            self.scores.append(np.mean(self.scores_deque))
            if episode % self.print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_deque)))
            if np.mean(self.scores_deque)>=90.0:
                print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(episode-100, np.mean(self.scores_deque)))
                break

