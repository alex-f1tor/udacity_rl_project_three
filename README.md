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
## Software requirements


The following python3 libraries are required:

`numpy == 1.16.2`

`pytorch == 0.4.0` - (GPU enabled)

`unity ML-agent` - available at [github](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

---

## Code implementation

This [notebook](https://github.com/alex-f1tor/udacity_rl_project_three/blob/master/DDPG/Tennis.ipynb) contains full pipeline of training networks:

* Initialization a unity environment;
* Initialization *Replay buffer* and *Agents* determined in [ddpg_collab_agent.py](https://github.com/alex-f1tor/udacity_rl_project_three/blob/master/DDPG/ddpg_collab_agent.py)
* Initialization *Actor* and *Critic* neural networks determined in [model.py](https://github.com/alex-f1tor/udacity_rl_project_three/blob/master/DDPG/model.py)
* Training and saving neural networks models at [models](https://github.com/alex-f1tor/udacity_rl_project_three/tree/master/DDPG/models) folder.