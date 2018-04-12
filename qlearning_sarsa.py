### Model free learning using Q-learning and SARSA
### You must not change the arguments and output types of the given function. 
### You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *
import matplotlib.pyplot as plt

def QLearning(env, num_episodes, gamma, lr, e):
    """Implement the Q-learning algorithm following the epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function
    num_episodes: int 
      Number of episodes of training.
    gamma: float
      Discount factor. 
    learning_rate: float
      Learning rate. 
    e: float
      Epsilon value used in the epsilon-greedy method. 


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """
    ############################
    #         YOUR CODE        #
    ############################
    def epsilon_greedy_policy(Q_one_state, epsilon, nA):
      probs = np.ones(nA, dtype=float) * epsilon / nA
      best_action = np.argmax(Q_one_state)
      probs[best_action] += (1.0 - epsilon)
      return probs

    Q = np.zeros([env.nS,env.nA])
    episodes_reward = np.zeros(num_episodes, dtype = float)
    average_reward = np.zeros(num_episodes, dtype = float)
    episode_length = np.zeros(num_episodes, dtype = float)

    for episode in range(num_episodes):
      state = env.reset()
      action_probs = epsilon_greedy_policy(Q[state], e, env.nA)
      action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

      while True:
        next_state, reward, terminal, _ = env.step(action)
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action] 
        td_delta = td_target - Q[state][action]
        Q[state][action] += lr * td_delta

        state = next_state
        action = best_next_action
        episodes_reward[episode] += reward
        episode_length[episode] += 1

        if terminal:
          break
    average_reward = np.cumsum(episodes_reward) / range(1,num_episodes+1)
    # plt.title("Q-Learning")
    # plt.xlabel("Episode")
    # plt.ylabel("Number of steps for each episode")
    # plt.yticks(np.arange(0,1400, 200))
    # plt.plot(range(1,num_episodes+1),episode_length)
    # plt.show()
    # plt.title("Q-Learning")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumu. Reward of Episode")
    # plt.plot(range(1,num_episodes+1),average_reward)
    # plt.show()
    # print("Q_Learning")
    # np.set_printoptions(threshold=np.nan)
    # print(Q)
    return Q
    # return np.zeros((env.nS, env.nA))


def SARSA(env, num_episodes, gamma, lr, e):
    """Implement the SARSA algorithm following epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function 
    num_episodes: int 
      Number of episodes of training
    gamma: float
      Discount factor. 
    learning_rate: float
      Learning rate. 
    e: float
      Epsilon value used in the epsilon-greedy method. 


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    ############################
    #         YOUR CODE        #
    ############################
    def epsilon_greedy_policy(Q_one_state, epsilon, nA):
      probs = np.ones(nA, dtype=float) * epsilon / nA
      best_action = np.argmax(Q_one_state)
      probs[best_action] += (1.0 - epsilon)
      return probs
      
    Q = np.zeros([env.nS,env.nA])
    episodes_reward = np.zeros(num_episodes, dtype = float)
    average_reward = np.zeros(num_episodes, dtype = float)
    episode_length = np.zeros(num_episodes, dtype = float)

    for episode in range(num_episodes):
      state = env.reset()
      action_probs = epsilon_greedy_policy(Q[state], e, env.nA)
      action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

      while True:
        next_state, reward, terminal, _ = env.step(action)

        next_action_probs = epsilon_greedy_policy(Q[next_state], e, env.nA)
        next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

        td_target = reward + gamma * Q[next_state][next_action] 
        td_delta = td_target - Q[state][action]
        Q[state][action] += lr * td_delta

        state = next_state
        action = next_action
        episodes_reward[episode] += reward
        episode_length[episode] += 1

        if terminal:
          break

    average_reward = np.cumsum(episodes_reward) / range(1,num_episodes+1)
    # plt.title("SARSA")
    # plt.xlabel("Episode")
    # plt.ylabel("Number of steps for each episode")
    # plt.yticks(np.arange(0,1400, 200))
    # plt.plot(range(1,num_episodes+1),episode_length)
    # plt.show()
    # plt.title("SARSA")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumu. Reward of Episode")
    # plt.plot(range(1,num_episodes+1),average_reward)
    # plt.show()
    # print("SARSA")
    # np.set_printoptions(threshold=np.nan)
    # print(Q)
    return Q
    # return np.ones((env.nS, env.nA))


def render_episode_Q(env, Q):
    """Renders one episode for Q functionon environment.

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play Q function on. 
      Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)  
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print ("Episode reward: %f" %episode_reward)



def main():
    env = gym.make("Assignment1-Taxi-v2")
    Q_QL= QLearning(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    Q_Sarsa = SARSA(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    # list(Q_QL.flat)
    # list(Q_Sarsa.flat)
    # np.set_printoptions(threshold=np.nan)
    # print(Q_QL)
    # print(Q_Sarsa)

    # render_episode_Q(env, Q_QL)
    # render_episode_Q(env, Q_Sarsa)
    # plt.title("Q-Learning")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumu. Reward of Episode")
    # plt.plot(range(1,1000+1),QL_reward)
    # plt.show()

    # plt.title("SARSA")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumu. Reward of Episode")
    # plt.plot(range(1,1000+1),SARSA_reward)
    # plt.show()

    # plt.title("Q-Learning and SARSA")
    # plt.xlabel("Episode")
    # plt.ylabel("Cumu. Reward of Episode")
    # plt.plot(range(1,1000+1),QL_reward, label = "Q-Learning")
    # plt.plot(range(1,1000+1),SARSA_reward, label = "SARSA")
    # plt.legend()
    # plt.title("Q-Learning and SARSA")
    # plt.xlabel("Episode")
    # plt.ylabel("Number of steps for each episode")
    # plt.plot(range(1,1000+1),Q_steps, label = "Q-Learning")
    # plt.plot(range(1,1000+1),SARSA_steps, label = "SARSA")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
