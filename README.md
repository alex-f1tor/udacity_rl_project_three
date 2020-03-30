# udacity_rl_project_three
udacity reinforcement learning project 3: Collaboration and Competition.

## The Environment

For this project, student have to train an 2 agents to playing tennis.

![Image](https://github.com/alex-f1tor/udacity_rl_project_three/blob/master/imgs/Tennis%20Collab%20DEMO.png)

A **reward** of +0.1 receives by an agent if ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The **state space** consists of 8 variables corresponding to the position and velocity of the ball and racket. 

Each **action** is a vector with 2 numbers, corresponding to movement toward (or away from) the net, and jumping. Every entry in the action vector must be a number between -1 and 1.

The task is episodic, and in order to solve the environment, agent must get an average score of +0.5 over 100 consecutive episodes.

---

The following python3 libraries are required:

`numpy == 1.16.2`

`pytorch == 0.4.0` - (GPU enabled)

`unity ML-agent` - available at [github](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

---

## Actor & Critic Networks

To solve the problem both actors were set by multilayer (2 hidden layers) dense neural networks were trained:

* Actor Networks: *input* - state vector, *hidden* - two layers 80 neurons, *output* - action 2-values vector;
* Critic Network: *input* - state & action vectors, *hidden* - two layers 24 & 48 neuron, *output* - advantage scalar value.

As part of the project, the task was solved using the following algorithms:

* DDPG - Deep Deterministic Policy Gradient [notebook](https://github.com/alex-f1tor/udacity_rl_project_three/blob/master/DDPG/Tennis.ipynb)


## Results

DDPG allows to get solution for each taining session, but the different number of episodes required:

![Image](https://github.com/alex-f1tor/udacity_rl_project_three/blob/master/imgs/ddpg_collab.png)



## Future work

* *Fine tune*: choose the optimal parameters for the noise mugnitude, learning rate, discount factor (gamma), soft update magnitude parameter (tau).

* *Wider and deeper NNs*: use neural networks with more neurons in hidden layers and more hidden layers.

* *Change stratagy*: use another algorithms for training agents: A3C, TRPO, PPO, Neuroevolution approach(https://arxiv.org/abs/1712.06567).





