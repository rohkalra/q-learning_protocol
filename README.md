# Q-Learning Protocol
Creating A Basic Q-Learning RL Model:

The code in this repository can be used as a foundation for your Q-Learning RL model. It contains clasess for both the environment and the agent.

In this task, there is a path that has been segmented into 10 different pieces. These segments are numbered 0-9 from left to right. On each trial, a reward is placed at a random location within [3, 7]. 

An RL agent is placed at location 0 to start, and then over the course of 1 learning episode (10,000 trials), the agent learns to run no matter what. This makes sense, as the agent learns that it will be rewarded every single time no matter what.

After running the example code, a bar-graph of learned q-values is output. More green represents that the agent learned to "run" when in that location. Alternatively, the agent learned to "stop" in a given location if that location's bar graph is majority red.
