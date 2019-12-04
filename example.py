# Author: Rohan Kalra
# Date: November 2019

#############################################################################################

# Import necessary packages

import sys
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt

#############################################################################################

# Environment specific information
nStates = 10
nActions = 2

# Agent specific information
aLearningRate = .1
aDiscount = .95
aExplorationRate = .1
aTrials = 10000

# Other information
RUN = 0
STOP = 1

#############################################################################################
# Class for RL environment

class Environment:
    def __init__(self):
        self.state = 1  # Default state (all the way to the left)
        self.location = -999 # Default reward location (non-existent)

    def determine_reward_location(self):
            self.location = np.random.randint(low=3, high=8)

    def take_action(self, action):
        if action == STOP:
            reward = 0
            self.state = 0
            return self.state, reward
        else:
            self.state+=1
            if self.state == self.location:
                reward = 1
                self.state = 0
            else:
                reward = 0
            return self.state, reward

#############################################################################################

# Class for RL Agent

class Agent:
    def __init__(self, learning_rate=aLearningRate, discount=aDiscount, exploration_rate=aExplorationRate, trials=aTrials):
        self.learning_rate = aLearningRate # How much we appreciate new q-value over current
        self.discount = aDiscount # How much we appreciate future reward over current
        self.exploration_rate = aExplorationRate # Initial exploration rate
        self.trials = aTrials # Number of Trials per learning episode
        self.q_table = np.zeros((nActions, nStates)) # Spreadsheet (Q-table) for rewards accounting

    def get_next_action(self, belief_state):
        if random.random() > self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(belief_state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        # Is RUN reward bigger?
        if self.q_table[RUN][state] > self.q_table[STOP][state]:
            return RUN
        # Is STOP reward bigger?
        elif self.q_table[STOP][state] > self.q_table[RUN][state]:
            return STOP
        # Rewards are equal, take random action
        return RUN if random.random() < 0.5 else STOP

    def random_action(self):
        return RUN if random.random() < 0.5 else STOP

    def update(self, old_state, new_state, action, reward):
        # Old Q-table value
        old_value = self.q_table[action][old_state]
        # Select next best action...
        future_action = self.greedy_action(new_state)
        # What is reward for the best next action?
        future_reward = self.q_table[future_action][new_state]

        # Main Q-table updating algorithm
        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        self.q_table[action][old_state] = new_value

#############################################################################################

# Misc. Methods

def line_graph(data):
    plt.figure(figsize=(15,15))
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]
    plt.plot(x_val,y_val)
    plt.plot(x_val,y_val,'or')
    plt.show()   

def bar_graph(labels, values1, values2):
    plt.figure(figsize=(15,15))
    # set width of bar
    barWidth = 0.25
     
    # set height of bar
    bars1 = values1
    bars2 = values2
     
    # Set position of bar on X axis
    r1 = np.arange(len(values1))
    r2 = [x + barWidth for x in r1]
     
    # Make the plot
    plt.bar(r1, bars1, color='#008000', width=barWidth, edgecolor='white', label='RUN')
    plt.bar(r2, bars2, color='#FF0000', width=barWidth, edgecolor='white', label='STOP')
     
    # Add xticks on the middle of the group bars
    plt.xlabel('State', fontweight='bold')
    plt.title('Probability of Each Action at Each State')

    # Create legend & Show graphic
    plt.legend()
    plt.show()

def convert_q2p(l, beta):
    to_ret = list()
    for pair in l:
        run = math.exp(pair[0]*beta)
        stop = math.exp(pair[1]*beta)
        s = run + stop
        run_p = run/s
        stop_p = stop/s
        to_ret.append((run_p, stop_p))
    return to_ret

#############################################################################################

def main():
    
    # breakpoint()
    prob_bottom = [0]


    # setup environment
    env = Environment()
    env.determine_reward_location()
    
    # setup agent
    agent = Agent()

    total_reward = 0 # Score keeping
    reward_over_time = list()
    locations = []

    # main loop
    count = 0
    while count < agent.trials:
        old_state = env.state
        action = agent.get_next_action(old_state)
        new_state, reward = env.take_action(action)
        agent.update(old_state, new_state, action, reward)

        if old_state != 1 and new_state == 1:
            count += 1
            print('Trial {} finished!'.format(count))
            env.determine_reward_location()

        total_reward += reward
        reward_over_time.append((count, total_reward))
    
    # Graph the Results

    run_qvalues = agent.q_table[0,:].tolist()
    stop_qvalues = agent.q_table[1,:].tolist()
    l = list(zip(run_qvalues, stop_qvalues))
    l = convert_q2p(l, 100)
    res = [[ i for i, j in l ], 
       [ j for i, j in l ]]
    run_p_values = res[0]
    stop_p_values = res[1]

    line_graph(reward_over_time)
    list_of_states = list(range(nStates))
    bar_graph(list_of_states, run_p_values, stop_p_values)

#############################################################################################

if __name__ == "__main__":
    main()

#############################################################################################

# END OF FILE
