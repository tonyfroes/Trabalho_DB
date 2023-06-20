from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from random import randint, seed
from matplotlib.pyplot import plot, show, subplots
from matplotlib.axes import *
import matplotlib.pyplot as plt

# Q- Learning agent
from collections import defaultdict

class PathFURG(Env):

  def __init__(self):
    self.action_space = Discrete(4) # 0 = R, 1 = D, 2 = L, 3 = U
    self.observation_space = Box(low=np.array([0]), high=np.array([99])) # [0 .. 15]
    self.state = 0 #Começa no 0
    self.red = [3,13,23,33,36,43,46,53,56,63,66,73,76,86,96,37,58,59,60,77,78] # Parede Vermelha
    self.bar = [39,81] # Bar
    self.rubi = [28,85] # Biblioteca
    self.goal = 99 # Matriz de 10x10
    self.rounds = 500 # Número de rodadas
    self.collected_reward = 0 # Recompensa coletada

  def toLinCol(self, st): 
    lin = st // 10
    col = st % 10
    return lin, col

  def fromLinCol(self, lin, col):
    return lin * 10 + col

  def reset(self):
    self.rounds = 500
    self.state = 0
    return self.state

  def step(self, action):
    lin, col = self.toLinCol(self.state)

    if (action == 0):
      col += 1
    elif (action == 1):
      lin += 1
    elif (action == 2):
      col -= 1
    else:
      lin -= 1

    reward = 0
    done = False
    self.rounds -= 1
    # verifica se esta dentro do grid
    if (col >= 0 and col <= 9 and lin >= 0 and lin <= 9):
      # esta no grid
      # altera o estado
      self.state = self.fromLinCol(lin, col)
      if (self.state == self.goal):
        reward = 50
        self.collected_reward += 50
        done = True

      elif (self.state in self.red):
         reward = -50
         self.collected_reward -= 50
         done = False

      elif (self.state in self.bar):
         reward = -30
         self.collected_reward -= 30
         done = False

      elif (self.state in self.rubi):
         reward = 0
         done = False

      else:
         reward = -1
         self.collected_reward -= 1
         done = False

    else:
      reward = -10
      self.collected_reward -= 10
      done = False

    if (self.rounds == 0):
      done = True

    self.render(action, reward)

    return self.state, reward, done, {}

  def render(self, action, reward):
    pass

env = PathFURG()

done = False

while not done:
  state = env.reset()
  action = env.action_space.sample()
  state = env.step(action)
  reward = env.step(action)
  done = env.step(action)
  info = env.step(action)

class QLearningAgent:
    def __init__(self, alpha, epsilon, gamma, get_possible_actions):
        self.get_possible_actions = get_possible_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self._Q = defaultdict(lambda: defaultdict(lambda: 0))

    def get_Q(self, state, action):
        return self._Q[state][action]

    def set_Q(self, state, action, value):
        self._Q[state][action] = value

    # Q learning update step
    def update(self, state, action, reward, next_state, done):
        if not done:
            best_next_action = self.max_action(next_state)
            td_error = reward + \
                       self.gamma * self.get_Q(next_state, best_next_action) \
                       - self.get_Q(state, action)
        else:
            td_error = reward - self.get_Q(state, action)

        new_value = self.get_Q(state, action) + self.alpha * td_error
        self.set_Q(state, action, new_value)

    # get best A for Q(S,A) which maximizes the Q(S,a) for actions in state S
    def max_action(self, state):
        actions = self.get_possible_actions(state)
        best_action = []
        best_q_value = float("-inf")

        for action in actions:
            q_s_a = self.get_Q(state, action)
            if q_s_a > best_q_value:
                best_action = [action]
                best_q_value = q_s_a
            elif q_s_a == best_q_value:
                best_action.append(action)
        return np.random.choice(np.array(best_action))

    # choose action as per epsilon-greedy policy for exploration
    def get_action(self, state):
        actions = self.get_possible_actions(state)

        if len(actions) == 0:
            return None

        if np.random.random() < self.epsilon:
            a = np.random.choice(actions)
            return a
        else:
            a = self.max_action(state)
            return a

# plot rewards
def plot_rewards(env_name, rewards, label):
    plt.title("env={}, Mean reward = {:.1f}".format(env_name,
                                                    np.mean(rewards[-20:])))
    plt.plot(rewards, label=label)
    plt.grid()
    plt.legend()
    plt.ylim(-300, 300)
    plt.show()

# training algorithm
def train_agent(env, agent, episode_cnt=500, tmax=10000, anneal_eps=True):
    episode_rewards = []
    for i in range(episode_cnt):
        G = 0
        state = env.reset()
        for t in range(tmax):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            G += reward
            if done:
                episode_rewards.append(G)
                # to reduce the exploration probability epsilon over the
                # training period.
                if anneal_eps:
                    agent.epsilon = agent.epsilon * 0.99
                break
            state = next_state
    return np.array(episode_rewards)

# helper fucntion to print policy for Cliff world
def print_policy(env, agent):
    #nR, nC = env._cliff.shape
    nR = 10
    nC = 10

    actions = '>v<^'

    for x in range(nR):
        for y in range(nC):
            pos = env.fromLinCol(x, y)
            if (pos == env.goal):
              print(" G ", end='')
            else:
                print(" %s " % actions[agent.max_action(pos)], end='')
        print()

env = PathFURG()

# Teste do ambiente
print(env.step(0))
print(env.step(0))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(3))
print(env.step(3))
print(env.step(3))
print(env.step(3))
print(env.step(3))
print(env.step(3))
print(env.step(0))
print(env.step(0))
print(env.step(0))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(1))
print(env.step(0))

# create a Q Learning agent
agent = QLearningAgent(alpha=0.1, epsilon=0.8, gamma=1,
                       get_possible_actions=lambda s : range(env.action_space.n))

#train agent and get rewards for episodes
rewards = train_agent(env, agent, episode_cnt=1000)

# Plot rewards
plot_rewards("Caminho até a FURG",rewards, 'Q-Learning') # Mostra o gráfico de recompensas

print_policy(env, agent)

agent._Q
