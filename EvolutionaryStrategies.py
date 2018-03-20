import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import datetime
import time

from ESN.ESN import LayeredESN, LCESN, EESN, ESN, DHESN
from Helper.utils import nrmse, LivePlotHistogram, LiveDataGraph

MAX_REWARD = 1000000

class GeneticAlgorithm(object):

    def __init__(self, reward_function, num_params, params_base=None, num_resamples=1,
                population=20, mutation_prob=0.2, mutation_scale=0.1, 
                selection_strategy='ranked', # roulette, ranked
                generation_update_strategy='elitism', # elitismWR (With Replacement), elitism, reset
                crossover_rule='two-uniform', # two-parent, single-parent, two-uniform
                verbose=False, seed=None):
        '''
        reward_function:    objective function to MAXIMISE
        num_params:         number of features in the model vector
        population:         number of individuals to create per generation
        base_run_rate:      number of episodes before we run the base parameters on the 
                                objective function
        '''

        if seed is not None: np.random.seed(seed)

        self.reward_function = reward_function
        self.num_params = num_params

        self.num_resamples = num_resamples
        self.population = population * self.num_resamples     # number of individuals in one population
        self.individuals = np.clip(np.random.randn(self.population, self.num_params)*0.0 +
        #np.array([0.85, 1.25, 1.0]), 0, 1.5) # the population, itself
        params_base, 0, 1.5)
        print("BASE PARAMETERS: {}".format(params_base))
        print("INITIAL POPULATION:")
        for idx,p in enumerate(self.individuals):
            print("\t {}: {}".format(idx, p))
        self.individuals_fitness = np.zeros(self.population)  # the fitness of each member of the population

        self.culm_mean_reward = None
        self.culm_mean_reward_base = None
        self.reward_hist_pop = []             # list of cumulative mean rewards for each step of optimization
        self.reward_hist_base = []

        self.mutation_prob = mutation_prob
        self.mutation_scale = mutation_scale
        self.selection_strategy = selection_strategy
        self.generation_update_strategy = generation_update_strategy
        self.crossover_rule = crossover_rule

        self.verbose = verbose
        self.base_run_rate = 1 
        self.SAVE_RATE = 4

        self.best_individual = np.zeros(self.num_params)
        self.params_base = params_base

        self.lph_pop = None
        self.lph_fitness = None
        self.lp_best = None
        self.lp_pop = None
        self.init_live_plotting()

    def init_live_plotting(self):
        def get_gene(i):
            return self.individuals[:, i]
        def get_echo(): 
            return get_gene(0)
        def get_spectral():
            return get_gene(1)
        def get_weightin():
            return get_gene(2)
        
        def get_fitness():
            return self.individuals_fitness

        def get_best_hist():
            d = [0]*2+self.reward_hist_base
            return range(len(d)), d
        def get_pop_hist():
            d = [0]*2+self.reward_hist_pop
            return range(len(d)), d

        self.lph_pop = LivePlotHistogram([get_echo, get_spectral, get_weightin], title="POPULATION (LIVE)",
                    names=["ECHO", "SPECT", "WEIGHTIN"])
        # plotted twice for hack. Should add another thing for data evaluation anyway
        self.lph_fitness = LivePlotHistogram([get_fitness, get_fitness], title="FITNESS (LIVE)",
                    names=["FITNESS", "FITNESS"])

        self.lp_best = LiveDataGraph(get_best_hist, title="BEST FITNESS (LIVE)")
        self.lp_pop = LiveDataGraph(get_pop_hist, title="AVG. POPULATION FITNESS (LIVE)")

    def sample_population(self):
        """ Form a new population/generation from the old generation. """
        new_generation = np.zeros((self.population, self.num_params))

        # invert the fitnesses because then smaller values are BETTER
        if self.selection_strategy == 'roulette':
            inv_fit = 1./self.individuals_fitness
            fit_norm = inv_fit / np.sum(inv_fit)
            pop_probs = np.cumsum(fit_norm)

        elif self.selection_strategy == 'ranked':
            # sort based on fitness
            idx_sort = self.individuals_fitness.argsort()
            self.individuals = self.individuals[idx_sort, :]
            # exponential probability selection
            inv_fit = 1./np.array(range(1, self.population))
            fit_norm = inv_fit / np.sum(inv_fit)
            pop_probs = np.cumsum(fit_norm)
        else:
            raise NotImplementedError('no selection strategy other than roulette/ranked implemented.')

        if self.verbose:
            print('selection probs: {}'.format(pop_probs))

        for k in range(0, self.population, self.num_resamples):
            if self.crossover_rule == 'two-parent':
                # sample an individual using the roulette strategy
                p1 = np.argwhere(pop_probs > np.random.rand())[0][0]  # first parent
                p2 = np.argwhere(pop_probs > np.random.rand())[0][0]  # second parent

                # cross-over =========================================
                # using single-point technique
                c = int(np.random.rand() * self.num_params)   # crossover point is chosen u.a.r. from [0, num_params-1]
                p = np.zeros(self.num_params)                 # child chromosome
                p[:c] = self.individuals[p1, :c]              # ^
                p[c:] = self.individuals[p2, c:]              # ^

                if self.verbose:
                    print('two-p crossover--> p1: {}, p2: {} at {} = {}'.format(p1, p2, c, p))
            elif self.crossover_rule == 'two-uniform':
                # sample an individual using the roulette strategy
                p1 = np.argwhere(pop_probs > np.random.rand())[0][0]  # first parent
                p2 = np.argwhere(pop_probs > np.random.rand())[0][0]  # second parent

                # cross-over =========================================
                # using single-point technique
                c = np.random.rand(self.num_params > 0.5).astype(float)   # crossover point is chosen u.a.r. from [0, num_params-1]
                p = np.zeros(self.num_params)                 # child chromosome
                p += c * self.individuals[p1, :]              # ^
                p += (1.0 - c) * self.individuals[p1, :]              # ^

                if self.verbose:
                    print('two-p crossover--> p1: {}, p2: {} at {} = {}'.format(p1, p2, c, p))
            elif self.crossover_rule == 'single-parent':
                p1 = np.argwhere(pop_probs > np.random.rand())[0][0]
                # child is a small perturbation of the parent
                p = p1 + np.random.randn(self.num_params)*0.01

                if self.verbose:
                    print('single-p crossover--> p1: {} = {}'.format(p1, p))

            # mutation ===========================================
            # each gene/nucleotide has a small probability of mutating with some Gaussian white noise
            mutated_idx = (np.random.rand(self.num_params) < self.mutation_prob).astype(int)
            mutation = mutated_idx * np.random.randn(self.num_params) * self.mutation_scale
            p += mutation
            # p = np.clip(p + mutation, 0., 2.)
            clip_rate = int(self.num_params / 3.)
            p[:clip_rate] = np.clip(p[:clip_rate], 0., 1.)
            p[clip_rate:clip_rate*2] = np.clip(p[clip_rate:clip_rate*2], 0., 1.5)
            p[clip_rate*2:clip_rate*3] = np.clip(p[clip_rate*2:clip_rate*3], 0., 1.5)

            if self.verbose:
                print('mutation ({})--> p: {}'.format(mutation, p))

            new_generation[k:(k+self.num_resamples), :] = p

        if self.generation_update_strategy == 'reset':
            pass
        elif self.generation_update_strategy == 'elitismWR':
            # keep one-third the parents
            for k in range(0, self.population/3, self.num_resamples):
                if self.selection_strategy == 'roulette':
                    p = np.argwhere(pop_probs > np.random.rand())[0][0]
                    if self.verbose:
                        print("keeping {} from old gen.".format(p))
                    new_generation[k, :] = self.individuals[p, :] 
        elif self.generation_update_strategy == 'elitism':
            idx_sort = self.individuals_fitness.argsort()
            for k in range(0, self.population/5, self.num_resamples):
                p_idx = idx_sort[k]
                new_generation[k, :] = self.individuals[p_idx, :]
        
        # completely replace the old generation
        self.individuals = np.array(new_generation)

    def step(self):
        """ Calculates fitness of each member of current population, then spawns a new generation. """
        # run the population through the reward function
        mean_reward = 0.0
        rewards = []
        pop_count = 0
        for k in range(np.shape(self.individuals)[0]):
            # run the individual through the objective function
            r, r_std = self.reward_function(self.individuals[k, :])

            if self.verbose:
                print("INDIVIDUAL {} --> reward: {} +- {} \n\t\t PARAMS: \n\tE:{}, \n\tS:{}, \n\tW:{}".format(k, r, r_std, 
                self.individuals[k, :8],
                self.individuals[k, 8:16],
                self.individuals[k, 16:]))

            self.individuals_fitness[k] = r

            # Test for survival
            if r > -MAX_REWARD:
                mean_reward += r
                pop_count += 1

        # mean reward of all the behaviours of the pop.
        mean_reward /= pop_count

        # record cumulative mean reward
        if self.culm_mean_reward is None:
            self.culm_mean_reward = mean_reward
        else:
            self.culm_mean_reward = 0.9*self.culm_mean_reward + 0.1*mean_reward

        self.reward_hist_pop.append(self.culm_mean_reward)

        # update the plot of the population diversities
        if self.lph_pop is not None:
            self.lph_pop.run()
        if self.lph_fitness is not None:
            self.lph_fitness.run()
        if self.lp_pop is not None:
            self.lp_pop.run()
        if self.lp_best is not None:
            self.lp_best.run()

        # get the new generation
        self.sample_population()

        return mean_reward, pop_count

    def play(self):
        r,_ = self.reward_function(self.params_base)
        print("reward received: {}".format(r))
        return r

    def train(self, num_steps, name):
        # store the start time
        start_time_sec = time.time()

        # best base score so far (so we can save only the best model)
        best_base = -100000

        for i in range(num_steps):
            mean_reward, num_survived = self.step()

            # run the environment on the base parameters
            if i % self.base_run_rate == 0:
                self.best_individual = self.individuals[np.argmax(self.individuals_fitness), :]
                base_run, b_std = self.reward_function(self.best_individual)
                # record cumulative mean reward of the base params
                if self.culm_mean_reward_base == None:
                    self.culm_mean_reward_base = base_run
                else:
                    self.culm_mean_reward_base = 0.9*self.culm_mean_reward_base + 0.1*base_run

                self.reward_hist_base.append(self.culm_mean_reward_base)

                print('episode {}, base reward: {} +- {}, pop. reward: {}, pop. reward ov. time: {}, base reward ov. time: {}, num survived: {}'.format(
                    i, base_run, b_std, mean_reward, self.culm_mean_reward, self.culm_mean_reward_base, num_survived)
                )

            # save the state every 20 epochs (or the last epochs)
            if i % self.SAVE_RATE == 0 or i > (num_steps - 2):
                # save the MODEL
                try:
                    f = open(name+'_MODELpartial.pkl', 'wb')
                    pkl.dump(self.best_individual, f)
                    f.close()
                    print("MODEL saved.")
                except:
                    print('FAILED TO SAVE PARTIAL MODEL.')

                if self.culm_mean_reward_base >= best_base:
                    try:
                        f = open(name+'_MODEL_BESTpartial.pkl', 'wb')
                        pkl.dump(self.best_individual, f)
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
            # if len(self.reward_hist_pop) > -0.1 and self.reward_hist_pop[-1] > -0.1:
            #     std_10 = np.std(self.reward_hist_pop[-10:])
            #     if std_10 <= 0.3:
            #         print("ENDED DUE TO CONVERGENCE.")
            #         break

        print('WARNING: No convergence achieved.')


class EvolutionaryStrategiesOptimiser(object):

    def __init__(self, reward_function, num_params, params_base=None,
                 population=100, std=0.01, learn_rate=0.01, num_resamples=1,
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

        if seed is not None: np.random.seed(seed)

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
        self.SAVE_RATE = 4

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
            r, r_std = self.reward_function(p)

            if self.verbose:
                print("INDIVIDUAL {} --> reward: {} +- {} \n\t\t PARAMS: {}".format(idx, r, r_std, p))

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
                base_run, _ = self.reward_function(self.params_base)
                # record cumulative mean reward of the base params
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

            # # look at the last 10 updates and if they are within a std of 3, we have converged
            # if len(self.reward_hist_pop) > -0.1 and self.reward_hist_pop[-1] > -0.1:
            #     std_10 = np.std(self.reward_hist_pop[-10:])
            #     if std_10 <= 0.3:
            #         print("ENDED DUE TO CONVERGENCE.")
            #         break


class Agent(object):

    def __init__(self, data_train, data_val, MEAN_OF_DATA, base_esn):
        '''
        data_train : (X, y)
        data_val : (X, y)
        base_esn : ESN to run the ES on
        '''
        assert isinstance(base_esn, EESN) or isinstance(base_esn, LCESN) or isinstance(base_esn, ESN) or isinstance(base_esn, DHESN), "bad ESN type of {}".format(type(base_esn))

        self.data_train = data_train
        self.data_val = data_val
        self.base_esn = base_esn
        self.MEAN_OF_DATA = MEAN_OF_DATA
        
        # parameters (excluded the regulariser because it would just go to a huge value,
        #   maybe you can find a way to fix this.): 
        # ([echo params], [spectral radii], [input_scale])
        if isinstance(self.base_esn, ESN):
            self.num_params = 3
            self.params_base = np.ones((self.num_params), dtype=np.float)

            self.params_base = np.array([0.5, 1.0, 1.0])
        else:
            self.num_params = self.base_esn.num_reservoirs*3
            self.params_base = np.ones((self.num_params), dtype=np.float)
            # initial heuristic that spectral radius is 1 and echo param is 0.5 and weight in is 1
            self.params_base[:self.base_esn.num_reservoirs] = 0.5
            self.params_base[self.base_esn.num_reservoirs*2:]=1.0

    def params_to_model(self, params):
        '''
        Converts a feature vector of parameters (the 'chromosome')
        into an ESN model to run the reward function on.
        '''
        if isinstance(self.base_esn, EESN):
            echo_params = params[:self.base_esn.num_reservoirs]
            spec_params = params[self.base_esn.num_reservoirs:self.base_esn.num_reservoirs*2]
            weightin_params = params[self.base_esn.num_reservoirs*2:]
            esn = EESN(input_size=self.base_esn.getInputSize(), output_size=self.base_esn.getOutputSize(), num_reservoirs=self.base_esn.num_reservoirs,
                        reservoir_sizes=self.base_esn.reservoir_sizes, echo_params=echo_params, #self.base_esn.output_activation,
                        init_echo_timesteps=self.base_esn.init_echo_timesteps, regulariser=self.base_esn.regulariser, debug=self.base_esn.debug)
            esn.initialize_input_weights(scales=weightin_params.tolist())
            esn.initialize_reservoir_weights(spectral_scales=spec_params.tolist())
        elif isinstance(self.base_esn, LCESN):
            echo_params = params[:self.base_esn.num_reservoirs]
            spec_params = params[self.base_esn.num_reservoirs:self.base_esn.num_reservoirs*2]
            weightin_params = params[self.base_esn.num_reservoirs*2:]
            esn = LCESN(input_size=self.base_esn.getInputSize(), output_size=self.base_esn.getOutputSize(), num_reservoirs=self.base_esn.num_reservoirs,
                        reservoir_sizes=self.base_esn.reservoir_sizes, echo_params=echo_params, #self.base_esn.output_activation,
                        init_echo_timesteps=self.base_esn.init_echo_timesteps, regulariser=self.base_esn.regulariser, debug=self.base_esn.debug)
            esn.initialize_input_weights(scales=weightin_params.tolist())
            esn.initialize_reservoir_weights(spectral_scales=spec_params.tolist())
        elif isinstance(self.base_esn, DHESN):
            echo_params = params[:self.base_esn.num_reservoirs]
            spec_params = params[self.base_esn.num_reservoirs:self.base_esn.num_reservoirs*2]
            weightin_params = params[self.base_esn.num_reservoirs*2:]
            esn = DHESN(input_size=self.base_esn.getInputSize(), output_size=self.base_esn.getOutputSize(), num_reservoirs=self.base_esn.num_reservoirs,
                        reservoir_sizes=self.base_esn.reservoir_sizes, echo_params=echo_params, #self.base_esn.output_activation,
                        init_echo_timesteps=self.base_esn.init_echo_timesteps, 
                        regulariser=self.base_esn.regulariser, 
                        debug=self.base_esn.debug,
                        dims_reduce=(np.linspace(200, 50, len(self.base_esn.encoders)).astype(int).tolist()),
                # init_echo_timesteps=100, dims_reduce=(np.linspace(50, 200, n-1).astype(int).tolist()),
                        encoder_type='VAE')
            esn.initialize_input_weights(scales=weightin_params.tolist())
            esn.initialize_reservoir_weights(spectral_scales=spec_params.tolist(), sparsity=0.1)
            # print(self.base_esn.regulariser)
        else: #ESN
            echo_params = params[0]
            spec_params = params[1]
            weightin_params = params[2]
            esn = ESN(input_size=self.base_esn.getInputSize(), output_size=self.base_esn.getOutputSize(),
                        reservoir_size=self.base_esn.N, echo_param=echo_params, #self.base_esn.output_activation,
                        init_echo_timesteps=self.base_esn.init_echo_timesteps, regulariser=self.base_esn.regulariser, debug=self.base_esn.debug)
            esn.initialize_input_weights(scale=weightin_params)
            esn.initialize_reservoir_weights(spectral_scale=spec_params)

        return esn

    def run_episode2(self, params):
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
        if nrmse_err > 10:
            nrmse_err = MAX_REWARD

        return -nrmse_err

    def run_episode(self, params, num_runs=3):
        esn = self.params_to_model(params)

        errors = np.zeros(num_runs)

        for run_num in range(num_runs):
            esn.train(self.data_train[0], self.data_train[1])

            # Run generative test, calculate NRMSE
            y_pred = []
            u_n = self.data_val[0][0]
            for _ in range(len(self.data_val[1])):
                u_n = esn.forward(u_n)
                y_pred.append(u_n)

            y_pred = np.array(y_pred).squeeze()
            y_vals = self.data_val[1].squeeze()
            # print(np.hstack((y_pred[:, None], y_vals[:, None])))
            nr = nrmse(y_vals, y_pred, self.MEAN_OF_DATA)
            if nr > 1000000000: nr = 1000000000
            errors[run_num] = nr

        # for r in esn.reservoirs:
        #     print("E:{}, S:{}, W:{}, REG:{}, SIZE:{},  SPARSE:{}".format(r.echo_param, r.spectral_scale, r.input_weights_scale, esn.regulariser, r.N, r.sparsity))

        # plt.plot(range(len(y_pred)), y_pred, label="pred")
        # plt.plot(range(len(y_vals)), y_vals, label="true")
        # plt.legend()
        # plt.show()

        return -np.mean(np.log(errors)), np.std(np.log(errors))
        # Calculate reward =============================
        # failures = errors[np.where(errors >= 1.)[0]]
        # num_failures = len(failures)
        # failure_rate = float(num_failures) / num_runs

        # if num_failures != num_runs:
        #     sucs = errors[np.where(errors < 1.)[0]]
        #     nrmse_suc = np.mean(sucs)
        # else:
        #     nrmse_suc = 1.0

        # if num_failures != 0:
        #     nrmse_fail = np.mean(failures)
        #     nrmse_fail_saturated = []
        #     for l in [1, 0.01, 0.001, 0.0001]:
        #         sat_err = 1. / (1. + np.exp(-l*nrmse_fail))
        #         sat_err = (sat_err - 0.5) * 2.
        #         nrmse_fail_saturated.append(sat_err)
        #     nrmse_fail_saturated = np.mean(nrmse_fail_saturated)
        # else:
        #     nrmse_fail_saturated = 0.
        
        # # failure_rate, nrmse_success, nrmse_fail_saturated
        # out = (1. - failure_rate) * nrmse_suc 
        # out += failure_rate * nrmse_fail_saturated
        # return -out
        #return -np.mean([failure_rate, 1.1*nrmse_suc, nrmse_fail_saturated])


def RunES(episodes, name, population, std, learn_rate, 
            data_train, data_val, MEAN_OF_DATA, base_esn,
            params_base):
    '''
    Call this function to setup the 'agent' and the ES optimiser to then
    do the optimisation.
    '''
    agent = Agent(data_train, data_val, MEAN_OF_DATA, base_esn)
    e_op = EvolutionaryStrategiesOptimiser(
        agent.run_episode, agent.num_params, params_base,
        population, std, learn_rate, verbose=False, num_resamples=1)

    e_op.train(episodes, name)

    try:
        f = open(name+'_MODEL.pkl', 'wb')
        pkl.dump((e_op.params_base, agent.num_params, agent.layers, agent.network_type), f)
        f.close()
    except:
        print('FAILED TO SAVE MODEL:'+name)

    return e_op.reward_hist_pop

def RunGA(episodes, name, population, data_train, data_val, MEAN_OF_DATA, base_esn, params_base, verbose=False):
    '''
    Call this function to setup the 'agent' and the GA optimiser to then
    do the optimisation.
    '''
    name = "Results/"+name
    agent = Agent(data_train, data_val, MEAN_OF_DATA, base_esn)
    ga_op = GeneticAlgorithm(
        reward_function=agent.run_episode, 
        num_params=agent.num_params, params_base=params_base,
        population=population, verbose=True, num_resamples=1)

    ga_op.train(episodes, name)

    try:
        f = open(name+'_MODEL.pkl', 'wb')
        pkl.dump((e_op.params_base, agent.num_params, agent.layers, agent.network_type), f)
        f.close()
    except:
        print('FAILED TO SAVE MODEL:'+name)

    # return e_op.reward_hist_pop
