from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512       # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 2e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = [Actor(state_size, action_size, random_seed).to(device) for cnt in range(num_agents)]
        self.actor_target = [Actor(state_size, action_size, random_seed).to(device) for cnt in range(num_agents)]
        self.actor_optimizer = [optim.Adam(self.actor_local[cnt].parameters(), lr=LR_ACTOR) for cnt in range(num_agents)]

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        #self.noise = OUNoise(action_size, random_seed)
        self.noise = OUNoise((2, ), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
   
    
    def step(self, states, actions, rewards, next_states, dones, step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones) 

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            if step%(BATCH_SIZE/8):
                self.learn(experiences, GAMMA)
       
    
    def act(self, state, net_index, episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local[net_index].eval()
        with torch.no_grad():
            action = self.actor_local[net_index](state[net_index]).cpu().data.numpy()
            self.actor_local[net_index].train()
            #print(action.shape)
            if add_noise:
                action += np.exp(-episode/2000)*self.noise.sample()
           
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        for net_index in range(2):
            next_states_cloned = next_states.clone()
            rewards_cloned = rewards.clone()
            dones_cloned = dones.clone()
            actions_cloned = actions.clone()
            states_cloned = states.clone()


            actions_next = self.actor_target[net_index](next_states_cloned[:,net_index,:])
            actions_next_cloned = actions_next.clone()
            
            Q_targets_next = self.critic_target(next_states_cloned[:,net_index,:], actions_next_cloned)

            if net_index==0:
                Q_targets_one = rewards_cloned[:,net_index].view(BATCH_SIZE, 1) + (gamma * Q_targets_next * (1 - dones_cloned[:,net_index,:]))
                # Compute critic loss
                Q_expected_one = self.critic_local(states_cloned[:,net_index,:], actions_cloned[:,net_index,:])
            
                actions_pred = self.actor_local[net_index](states_cloned[:,net_index,:])
                
                actor_loss_one = -self.critic_local(states_cloned[:,net_index,:], actions_pred)

            else:
                Q_targets_two = rewards_cloned[:,net_index].view(BATCH_SIZE, 1) + (gamma * Q_targets_next * (1 - dones_cloned[:,net_index,:]))
                # Compute critic loss
                Q_expected_two = self.critic_local(states_cloned[:,net_index,:], actions_cloned[:,net_index,:])
               
                actions_pred = self.actor_local[net_index](states_cloned[:,net_index,:])
                                               
                actor_loss_two= -self.critic_local(states_cloned[:,net_index,:], actions_pred)

        
        Q_expected = torch.cat([Q_expected_one.view(1,BATCH_SIZE,1), Q_expected_two.view(1,BATCH_SIZE,1)]).mean(0)
        Q_targets = torch.cat([Q_targets_one.view(1,BATCH_SIZE,1), Q_targets_two.view(1,BATCH_SIZE,1)]).mean(0)
        actor_loss = torch.cat([actor_loss_one.view(1,BATCH_SIZE,1), actor_loss_two.view(1,BATCH_SIZE,1)]).mean()
        
        critic_loss = F.mse_loss(Q_expected.clone(), Q_targets.clone())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        self.soft_update(self.critic_local, self.critic_target, TAU)



        for net_index in range(2):
   
            
            self.actor_optimizer[net_index].zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer[net_index].step()


            
            self.soft_update(self.actor_local[net_index], self.actor_target[net_index], TAU) 



    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.35, sigma=0.5):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agents, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, np.array(reward), next_state, np.array(done))
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state.reshape(1,num_agents, -1) for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action.reshape(1,num_agents, -1) for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward.reshape(1, num_agents) for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state.reshape(1,num_agents, -1) for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done.reshape(1, num_agents, -1) for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

