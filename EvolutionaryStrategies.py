import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle as pkl

import datetime
import time

from ESN.ESN import LayeredESN, LCESN, EESN, ESN
from Helper.utils import nrmse

class EvolutionaryStrategiesOptimiser(object):

    def __init__(self, reward_function, num_params, params_base=None,
                 population=100, std=0.1, learn_rate=0.01, num_resamples=3,
                 seed=None,
                 verbose=False, base_run_rate=1):
        '''
        reward_function:    objective function to MAXIMISE
        num_params:         number of features in the model vector
        population:         number of individuals to create per generation
        std:                scale of the Gaussian Noise to apply to the parameters
        learn_rate:         rate of SGD
        num_resamples:      (added for ESN) number of times to add each individual so that 
                                we can remove effects of random weights.
        base_run_rate:      number of episodes before we run the base parameters on the 
                                objective function
        '''

        if seed is not None: np.random.seed(10)

        self.reward_function = reward_function
        self.num_params = num_params

        if len(params_base) == 0:
            self.params_base = np.zeros(num_params)
        else:
            self.params_base = params_base

        self.population = population
        self.std = std
        self.learn_rate = learn_rate
        self.num_resamples = num_resamples

        self.culm_mean_reward = None
        self.culm_mean_reward_base = None
        self.reward_hist_pop = []
        self.reward_hist_base = []

        self.verbose = verbose
        self.base_run_rate = 1 
        self.SAVE_RATE = 20

    def sample_population(self):
        # each individual in a population is a certain amount of noise which we add to the
        #  base individual. We use this noise to compute the gradient later
        pop = []
        for k in range(self.population):
            p = np.random.randn(self.num_params)
            for i in range(0, self.num_resamples):
                pop.append(p)

        return pop

    def step(self):
        # get a sample population of noise:
        pop = self.sample_population()

        # run the population through the reward function
        mean_reward = 0.0
        rewards = []
        for idx,k in enumerate(pop):
            # we add the noise to the base parameter and see how the perturbation
            #  affects the output (similar to finite gradients estimation)
            p = np.clip(self.params_base + k*self.std, 0., 2.)
            r = self.reward_function(p)

            if self.verbose:
                print("INDIVIDUAL {} --> reward: {} \n\t\t PARAMS: {}".format(idx, r, p))

            rewards.append(r)
            mean_reward += r

        # mean reward of all the behaviours of the pop.
        mean_reward /= self.population

        # record cummulative mean reward
        if self.culm_mean_reward == None:
            self.culm_mean_reward = mean_reward
        else:
            self.culm_mean_reward = 0.9*self.culm_mean_reward + 0.1*mean_reward

        self.reward_hist_pop.append(self.culm_mean_reward)

        # normalise the rewards (because we want our gradients to be normalised no
        #  matter how large our reward is)
        rewards -= np.mean(rewards)
        rewards /= (np.std(rewards))

        # gradient is compute by summing up the noise from each individual weighted
        #  by it's success (amount of reward)
        g = np.dot(rewards[None, :], np.array(pop)).squeeze()

        # recompute the mean of the parameters using the reward as the gradient
        # print('update grad: {}'.format(self.params_base))
        self.params_base += self.learn_rate/(self.population*self.std) * g

        self.params_base = np.clip(self.params_base, 0., 2.)

        return mean_reward

    def play(self):
        r,_ = self.reward_function(self.params_base)
        print("reward received: {}".format(r))
        return r

    def train(self, steps, name):

        # store the start time
        start_time_sec = time.time()

        # best base score so far (so we can save only the best model)
        best_base = -100000

        for i in range(steps):

            mean_reward = self.step()

            # run the environment on the base parameters
            if i % self.base_run_rate == 0:
                base_run = self.reward_function(self.params_base)
                # record cummulative mean reward of the base params
                if self.culm_mean_reward_base == None:
                    self.culm_mean_reward_base = base_run
                else:
                    self.culm_mean_reward_base = 0.9*self.culm_mean_reward_base + 0.1*base_run

                self.reward_hist_base.append(self.culm_mean_reward_base)

                print('episode {}, base reward: {}, pop. reward: {}, pop. reward ov. time: {}, base reward ov. time: {}'.format(
                i, base_run, mean_reward, self.culm_mean_reward, self.culm_mean_reward_base))

            # save the state every 20 epochs (or the last epochs)
            if i % self.SAVE_RATE == 0 or i > steps - 2:
                # save the MODEL
                try:
                    f = open(name+'_MODELpartial.pkl', 'wb')
                    pkl.dump(self.params_base, f)
                    f.close()
                    print("MODEL saved.")
                except:
                    print('FAILED TO SAVE PARTIAL MODEL.')

                if self.culm_mean_reward_base >= best_base:
                    try:
                        f = open(name+'_MODEL_BESTpartial.pkl', 'wb')
                        pkl.dump(self.params_base, f)
                        f.close()
                        print("!!BEST MODEL saved.")
                        best_base = self.culm_mean_reward_base
                    except:
                        print('FAILED TO SAVE BEST MODEL')

                # save the cummulative reward DATA
                try:
                    f = open(name+'_DATApartial.pkl', 'wb')
                    pkl.dump((self.reward_hist_pop, self.reward_hist_base), f)
                    f.close()
                    print("DATA saved.")
                except:
                    print('FAILED TO SAVE PARTIAL DATA.')

                # save the STATS
                try:
                    total_time_sec = time.time() - start_time_sec
                    total_time_min = float(total_time_sec) / 60.0
                    stats = "Total run time mins: {}.".format(total_time_min)

                    f = open(name+'_STATSpartial.pkl', 'wb')
                    pkl.dump(stats, f)
                    f.close()
                    print("STATS saved.")
                except:
                    print("FAILED TO SAVE PARTIAL STATS.")

            # look at the last 10 updates and if they are within a std of 3, we have converged
            if len(self.reward_hist_pop) > -0.1 and self.reward_hist_pop[-1] > -0.1:
                std_10 = np.std(self.reward_hist_pop[-10:])
                if std_10 <= 0.3:
                    print("ENDED DUE TO CONVERGENCE.")
                    break


class Agent(object):

    def __init__(self, data_train, data_val, MEAN_OF_DATA, base_esn):
        '''
        data_train : (X, y)
        data_val : (X, y)
        base_esn : ESN to run the ES on
        '''
        assert isinstance(base_esn, EESN) or isinstance(base_esn, LCESN) or isinstance(base_esn, ESN), "bad ESN type of {}".format(type(base_esn))

        self.data_train = data_train
        self.data_val = data_val
        self.base_esn = base_esn
        self.MEAN_OF_DATA = MEAN_OF_DATA
        
        # parameters (excluded the regulariser because it would just go to a huge value,
        #   maybe you can find a way to fix this.): 
        # ([echo params], [spectral radii], [input_scale])
        if isinstance(base_esn, ESN):
            self.num_params = 3
            self.params_base = np.ones((self.num_params), dtype=np.float)

            self.params_base = np.array([0.5, 1.0, 1.0])
        else:
            self.num_params = self.base_esn.num_reservoirs*3
            self.params_base = np.ones((self.num_params), dtype=np.float)
            # initial heuristic that spectral radius is 1 and echo param is 0.5 and weight in is 1
            self.params_base[:self.base_esn.num_reservoirs] = 0.5
            self.params_base[self.base_esn.num_reservoirs*2:-1]=1.0

    def params_to_model(self, params):
        '''
        Converts a feature vector of parameters (the 'chromosome')
        into an ESN model to run the reward function on.
        '''
        if isinstance(self.base_esn, EESN):
            echo_params = params[:self.base_esn.num_reservoirs]
            spec_params = params[self.base_esn.num_reservoirs:self.base_esn.num_reservoirs*2]
            weightin_params = params[self.base_esn.num_reservoirs*2:-1]
            esn = EESN(input_size=self.base_esn.getInputSize(), output_size=self.base_esn.getOutputSize(), num_reservoirs=self.base_esn.num_reservoirs,
                        reservoir_sizes=self.base_esn.reservoir_sizes, echo_params=echo_params, #self.base_esn.output_activation,
                        init_echo_timesteps=self.base_esn.init_echo_timesteps, regulariser=reg, debug=self.base_esn.debug)
            esn.initialize_input_weights(scales=weightin_params.tolist())
            esn.initialize_reservoir_weights(spectral_scales=spec_params.tolist())
        elif isinstance(self.base_esn, LCESN):
            echo_params = params[:self.base_esn.num_reservoirs]
            spec_params = params[self.base_esn.num_reservoirs:self.base_esn.num_reservoirs*2]
            weightin_params = params[self.base_esn.num_reservoirs*2:-1]
            esn = LCESN(input_size=self.base_esn.getInputSize(), output_size=self.base_esn.getOutputSize(), num_reservoirs=self.base_esn.num_reservoirs,
                        reservoir_sizes=self.base_esn.reservoir_sizes, echo_params=echo_params, #self.base_esn.output_activation,
                        init_echo_timesteps=self.base_esn.init_echo_timesteps, regulariser=reg, debug=self.base_esn.debug)
            esn.initialize_input_weights(scales=weightin_params.tolist())
            esn.initialize_reservoir_weights(spectral_scales=spec_params.tolist())
        else: #ESN
            echo_params = params[0]
            spec_params = params[1]
            weightin_params = params[2]
            esn = ESN(input_size=self.base_esn.getInputSize(), output_size=self.base_esn.getOutputSize(),
                        reservoir_size=self.base_esn.N, echo_param=echo_params, #self.base_esn.output_activation,
                        init_echo_timesteps=self.base_esn.init_echo_timesteps, regulariser=esn.regulariser, debug=self.base_esn.debug)
            esn.initialize_input_weights(scale=weightin_params)
            esn.initialize_reservoir_weights(spectral_scale=spec_params)

        return esn

    def run_episode(self, params):

        esn = self.params_to_model(params)

        esn.train(self.data_train[0], self.data_train[1])

        # run generative and check
        y_pred = []

        # GENERATIVE =================================================
        u_n_ESN = self.data_val[0][0]
        for _ in range(len(self.data_val[1])):
            u_n_ESN = esn.forward(u_n_ESN)
            y_pred.append(u_n_ESN)

        y_pred = np.array(y_pred).squeeze()
        y_vals = self.data_val[1].squeeze()
        nrmse_err = nrmse(y_vals, y_pred, self.MEAN_OF_DATA)

        # avoid explosions
        if nrmse_err > 10000:
            nrmse_err = 10000

        return -nrmse_err

def RunES(episodes, name, population, std, learn_rate, 
            data_train, data_val, MEAN_OF_DATA, base_esn):
    '''
    Call this function to setup the 'agent' and the ES optimiser to then
    do the optimisation.
    '''
    agent = Agent(data_train, data_val, MEAN_OF_DATA, base_esn)
    e_op = EvolutionaryStrategiesOptimiser(
        agent.run_episode, agent.num_params, agent.params_base,
        population, std, learn_rate)

    e_op.train(episodes, name)

    try:
        f = open(name+'_MODEL.pkl', 'wb')
        pkl.dump((e_op.params_base, agent.num_params, agent.layers, agent.network_type), f)
        f.close()
    except:
        print('FAILED TO SAVE MODEL:'+name)

    return e_op.reward_hist_pop

