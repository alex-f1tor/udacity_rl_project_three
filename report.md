# udacity_rl_project_three_report

## Actor & Critic Networks

To solve the problem both actors were set by multilayer (2 hidden layers) dense neural networks were trained:

* Actor Networks: *input* - state vector, *hidden* - two layers 80 neurons, *output* - action 2-values vector;
* Critic Network: *input* - state & action vectors, *hidden* - two layers 24 & 48 neuron, *output* - advantage scalar value.

As part of the project, the task was solved using the following algorithm:

* DDPG - Deep Deterministic Policy Gradient [arxiv](https://arxiv.org/pdf/1509.02971.pdf)

* Learning hyperparameters:
	* *gamma* (discount factor) = 0.95;
	* *tau* (soft update of target parameter) = 0.001;
	* *learning rate* = 0.0002 (actor network), 0.002 (critic network);
	* *buffer size* (size of total amount of replay buffer) = 100 000;
	* *mu, simga, theta* (Ornstein-Uhlenbeck noise parameters): mean = 0, standard deviation = 0.5, theta - multiplier of mean and measurement difference.



## Results

DDPG allows to get solution for each taining session, but the different number of episodes required:

![Image](https://github.com/alex-f1tor/udacity_rl_project_three/blob/master/imgs/ddpg_collab.png)



## Future work

* *Fine tune*: choose the optimal parameters for the noise mugnitude, learning rate, discount factor (gamma), soft update magnitude parameter (tau).

* *Wider and deeper NNs*: use neural networks with more neurons in hidden layers and more hidden layers.

* *Change stratagy*: use another algorithms for training agents: A3C, TRPO, PPO, Neuroevolution approach.





