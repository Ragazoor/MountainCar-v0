import numpy as np
from FFNN import *
import copy
from math import log, sqrt

# init

# optimise - takes and episode and optimises it

# Iterates through every episode and calls optimise


class FFNNAgent(object):
    def __init__(self, hyperparams):
        
        self.gamma = hyperparams['gamma']
        self.n_input_nodes = hyperparams['n_input_nodes']
        self.n_hidden_nodes = hyperparams['n_hidden_nodes']
        self.n_output_nodes= hyperparams['n_output_nodes']
        self.n_steps = hyperparams['n_steps']
        self.seed = hyperparams['seed']
    
        self.init_net_wr = hyperparams['init_net_wr']
        self.batch_size = hyperparams['batch_size']
        self.epsilon = hyperparams['epsilon']
        self.epsilon_min = hyperparams['epsilon_min']
        self.epsilon_decay_rate = hyperparams['epsilon_decay_rate']

        self.target_net_hold_episodes = hyperparams['target_net_hold_epsiodes']
        self.learning_rate = hyperparams['learning_rate']
        self.learning_rate_min = hyperparams['learning_rate_min']
        self.learning_rate_decay = hyperparams['learning_rate_decay']
        self.n_updates_per_episode = hyperparams['n_updates_per_episode']
        self.max_memory_len = hyperparams['max_memory_len']
        self.n_iter = hyperparams['n_iter']
        self.n_episodes_per_print = hyperparams['n_episodes_per_print']
        self.net_hold_eps = hyperparams['net_hold_epsilon']
        self.net_hold_lr = hyperparams['net_hold_lr']
        self.nmr_decimals_tiles = hyperparams['nmr_decimals_tiles']

        node_array = [self.n_input_nodes, self.n_hidden_nodes, self.n_output_nodes]        
        self.net = FFNN(node_array, self.learning_rate, self.seed, self.init_net_wr)
        self.net.initSession()

        self.memory = []
        self.reward_list = []
        np.random.seed(self.seed)

        # Monte Carlo Q-tree (That's a fantastic name!)
        self.tree_nodes = {} # Each nodes is reached with a key, a state + action
        self.C = hyperparams['C']
        self.score = {} # Score for each Q-state in tree
        self.plays = {} # Number of times a Q-state have been played 
        

    def optimize(self,sars_tuples, i_episode):
        self.memory += list(sars_tuples)
        if  len(self.memory) > self.max_memory_len:
            self.memory = self.memory[-self.max_memory_len:]
        memory_len = len(self.memory)

        if i_episode % self.target_net_hold_episodes == 0:
            self.target_net = copy.copy(self.net)


        if i_episode % self.net_hold_eps == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate
                print 'epsilon:',self.epsilon
        
        if i_episode % self.net_hold_lr == 0 and i_episode != 0:
            if self.learning_rate > self.learning_rate_min:
                self.learning_rate *= self.learning_rate_decay
                #self.net.set_lr(self.learning_rate)
                print 'learning_rate:',self.net.lr
                        
        if memory_len > self.batch_size:
            if i_episode < 1:
                n_updates = 1 # Don't overtrain on the first (inevitebly bad) episodes 
            else:
                n_updates = self.n_updates_per_episode
            for _ in range(n_updates):
                idx_memory_batch = np.random.choice(range(memory_len), size = self.batch_size, replace = False)

                states = []
                Q_target = []
                for idx_memory in idx_memory_batch:
                    s, a, r, s_prime, done = self.memory[idx_memory]
                    
                    all_targets = self.net.get_Q(np.matrix(s))[0]
                    if done:
                        all_targets[a] = r
                    else:
                        all_targets[a] = r + self.gamma * np.max(self.target_net.get_Q(np.matrix(s_prime)))
                    
                    states.append(s)
                    Q_target.append(all_targets)
                if i_episode % self.n_episodes_per_print == 0:
                    # PRINTS Q
                    print all_targets
                    
            self.net.gd(x_batch = np.asmatrix(states), Q_batch = np.asmatrix(Q_target))
            

    def round_2_tile(self, state):
        return np.round(state, decimals = self.nmr_decimals_tiles)


    def take_action(self, env, t, state, visited_states):
        action_list = range(self.n_output_nodes)
        state_cpy = state[:] # copy state
        state_cpy = self.round_2_tile(state_cpy) # round state 2 tile
        state_cpy = state_cpy.tolist() # matrix to list
        state_cpy = tuple(state_cpy) # list to tuple
    
        
        if all(self.plays.get((state_cpy,a)) for a in action_list):
            # Use neural net
            log_total = log(sum(self.plays[(state_cpy, a)] for a in action_list))
            Q = self.net.get_Q(np.matrix(state))[0]
            
            _, action = max((Q[a]+self.C*sqrt(log_total/ self.plays[(state_cpy,a)]),a) for a in action_list)
        else:
            action = env.action_space.sample()
        
        visited_states.append((state_cpy, action)) # add state action pair to visited states
        # Add the state to our plays/score thing
        if (state_cpy, action) not in self.plays:
            self.plays[(state_cpy, action)] = 0
            self.score[(state_cpy, action)] = 0

        return action


    def create_episode(self, env, i_episode, rend):
        done = False
        visited_states = []
        state = env.reset()
        sars_tuples = []
        t = 1
        tot_reward = 0
        while not done and t < self.n_steps+1:
            if i_episode %self.n_episodes_per_print == 0 and rend:
                env.render()            
            action = self.take_action(env, t, state, visited_states)

            sars = [state, action]
            state, r, done, info = env.step(action)
            sars += [r,state, done]
            sars_tuples.append(tuple(sars))
            t += 1
            tot_reward += r
        
        for state, a in visited_states:
            if (state,a) not in self.plays: # shouldn't do anything now, all states are saved
                continue
            self.plays[(state,a)] += 1
            self.score[(state,a)] += tot_reward
            if i_episode % self.n_episodes_per_print == 0:
                print 'action and plays for', state,':', a, self.plays[(state,a)]
        return sars_tuples, t, tot_reward


    def optimize_episodes(self, env, rend = False):
        t_avg = 0
        r_avg = 0
        for i_episode in range(self.n_iter):
            sars_tuples, t, tot_reward = self.create_episode(env, i_episode, rend)
            self.optimize(sars_tuples, i_episode)
            t_avg += t
            r_avg += tot_reward
            self.reward_list.append(tot_reward)
            if i_episode %self.n_episodes_per_print == 0:
                print 'Episode Length', t
                print 'Total reward', tot_reward
                print i_episode

        print 'Average Length :',t_avg/float(self.n_iter)
        print 'Average Reward :',r_avg/float(self.n_iter)

    def plot_reward(self):
        plt.plot([np.mean(self.reward_list[i-50:i]) for i in range(len(self.reward_list))])


