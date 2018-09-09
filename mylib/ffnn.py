from collections import OrderedDict
from copy import deepcopy
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt

import numpy as np


class FFNN(nn.Module):
    """
    Feedforward neural network for modelling (chaotic) time series data.
        (currently only works for 1-dimensional data e.g. MackeyGlass).

    Args:
        input_size:             Number of frames of context (data for previous time steps).
                                 (not to be confused with data dimensionality).
        hidden_size:            Number of hidden units per hidden layer.
        n_hidden_layers:        Number of hidden layers (not including input+output layers).
        activation:             PyTorch activation (class, NOT an instance)
    """

    def __init__(self, input_size, hidden_size, n_hidden_layers, activation=None):
        super(FFNN, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(input_size)
        self.n_hidden_layers = int(n_hidden_layers)

        if activation is None:
            activation = nn.Sigmoid
        else:
            assert type(activation) == type, "Pass the TYPE of activation, not an instance of it."
        self.activ_str = str(activation)[:-2]

        layers = OrderedDict()
        layers['linear1'] = nn.Linear(input_size, hidden_size) # input layer
        layers['activ1'] = activation()
        for i in range(2, n_hidden_layers+2):
            # add hidden layers
            k1, k2 = 'linear%d' % i, 'activ%d' % i
            layers[k1] = nn.Linear(hidden_size, hidden_size)
            layers[k2] = activation()

        out_key = 'linear%d' % (n_hidden_layers + 2)
        layers[out_key] = nn.Linear(hidden_size, 1) # output layer

        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)

def train(model, train_data, batch_size, num_epochs, criterion, optimizer, valid_data=None, 
          verbose=1, eval_gen_loss=False, n_generate_timesteps=2000):
    input_size = model.input_size
    #assert (len(train_data) - input_size) % batch_size == 0, \
    #            "there is leftover training data that doesn't fit neatly into a batch"

    n_iter = int((len(train_data) - input_size) / batch_size)

    # rows: epoch number. columns: (sup. train nrmse, sup. valid nrmse, gen. train nrmse, 
    #    gen. valid nrmse). If valid_data not provided, last 3 columns are zeros. 
    #    Else if eval_gen_loss=False, last two columns zeros.
    stats = np.zeros((num_epochs, 4))

    if eval_gen_loss:
        # 'early stopping': return the model that gives lowest validation generation NRMSE
        best_model = (None, np.inf, None)

    for epoch in range(num_epochs):
        
        train_loss = 0.
        for i in range(0, n_iter, batch_size):
            inputs = torch.FloatTensor(batch_size, input_size)
            targets = torch.FloatTensor(batch_size)
            for batch_idx, j in enumerate(range(i, i+batch_size)):
                # inputs[batch_idx] = torch.FloatTensor(train_data[j:(j+input_size)])
                inputs[batch_idx] = torch.FloatTensor(train_data[j:(j+input_size)])
                targets[batch_idx] = train_data[j+input_size]

            inputs = Variable(inputs)
            targets = Variable(targets)

            # fprop, bprop, optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # normalized root mean square error
            train_loss += nrmse(outputs, targets)**2

        train_loss = np.sqrt(train_loss)
        stats[epoch, 0] = train_loss
        if verbose:
            print('='*50)
            print('Epoch [%d/%d]' % (epoch+1, num_epochs))
            print('Total sup. training NRMSE: %.7f' % train_loss)

        # Calculate GENERATION training loss ======================================
        if eval_gen_loss:
            gen_outs, nrms_error = test(model, train_data[:(n_generate_timesteps+input_size)], 
                                        plot=False)
            print('Generation training NRMSE (for %d time steps): %.7f' % \
                    (n_generate_timesteps, nrms_error))
            stats[epoch, 2] = nrms_error

        if valid_data is not None:
            # Calculate SUPERVISED validation loss ================================
            outputs = []
            da_targets = []
            for i in range(len(valid_data) - input_size):
                inputs = valid_data[i:(i+input_size)]
                inputs = Variable(torch.FloatTensor(inputs))
                target = Variable(torch.FloatTensor([valid_data[i+input_size]]))
                da_targets.append(target)

                output = model(inputs)
                mse = criterion(output, target).data[0]
                outputs += [output]
           
            valid_loss = float(nrmse(outputs, da_targets))
            stats[epoch, 1] = valid_loss
            if verbose:
                print('Total sup. validation NRMSE: %.7f' % valid_loss)

            if eval_gen_loss:
                # Now calculate GENERATION validation loss ===========================
                gen_outs, nrms_error = test(model, valid_data[:(n_generate_timesteps+input_size)], 
                                            plot=False)
                print('Generation validation NRMSE (for %d time steps): %.7f' % \
                        (n_generate_timesteps, nrms_error))
                stats[epoch, 3] = nrms_error
                
                if nrms_error <= best_model[1]:
                    best_model = (deepcopy(model), nrms_error, epoch)
    
    if valid_data is not None and eval_gen_loss:
        print('BEST EPOCH: %d' % (best_model[2]+1))
        return best_model[0], stats
    else:
        return model, stats

def to_numpy(arr):
    if isinstance(arr, list) and isinstance(arr[0], Variable):
        arr = [o.data.numpy() for o in arr]
    if isinstance(arr, list) and isinstance(arr[0], torch.Tensor):
        arr = [o.numpy() for o in arr]
    if isinstance(arr, Variable):
        arr = arr.data
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()

    return np.array(arr).squeeze()

def nrmse(outputs, targets):
    """ 
    fuck dynamic typing, this takes any array-like things and returns an NRMSE.
    info: www.doc.ic.ac.uk/teaching/distinguished-projects/2013/j.forman-gornall.pdf 
            (^ ERROR there: pi in the denominator should be oi)
    """ 
    outputs = to_numpy(outputs)
    targets = to_numpy(targets)
    assert len(outputs.shape) == 1
    assert len(targets.shape) == 1

    # normalizer: square error if we just predicted the true mean
    numer = np.sum((outputs - targets)**2)
    # denom = np.sum((targets - np.mean(targets))**2) # <- 'true' mean: mean of argument 'targets'
    denom = np.sum((targets - __DATA_MEAN__)**2) # <- 'true' mean: mean of all training+test data

    # normalizer: variance of data
    #denom = __DATA_VAR__
    #numer = np.mean((outputs - targets)**2)

    return np.sqrt(numer / denom)

def calculate_rmse(outputs, targets):
    outputs = to_numpy(outputs)
    targets = to_numpy(targets)
    return np.sqrt(np.mean((outputs - targets)**2)) 

def test(model, data, sample_step=None, plot=True, show_error=True, save_fig=False, title=None):
    """ 
    Pass the trained model. 
    Returns (generated_outputs, generation_nrmse).
    """

    input_size = model.input_size

    inputs = data[:input_size] # type(inputs) = list
    output = model(Variable(torch.FloatTensor(inputs))).data[0]
    generated_data = [output]

    for i in range(input_size, len(data)-1):
        # every 'sample_step' iterations, feed the true value back in instead of generated value
        if sample_step is not None and (i % sample_step) == 0:
            inputs.extend([data[i]])
            inputs = inputs[1:]
        else:
            inputs.extend([output]) # shift input
            inputs = inputs[1:]     # data

        output = model(Variable(torch.FloatTensor(inputs))).data[0]
        generated_data.append(output)
    
    error = nrmse(generated_data, data[input_size:])
    rmse =  calculate_rmse(generated_data, data[input_size:])

    # print('MSE: %.7f' % error)
    if plot:
        xs = range(len(generated_data))
        f, ax = plt.subplots(figsize=(16, 10))
        if title is not None:
            ax.set_title(title+('; error=%.5f' % error))
        ax.plot(xs, data[input_size:], label='True data')
        ax.plot(xs, generated_data, label='Generated data')
        if sample_step is not None:
            smp_xs = np.arange(0, len(xs), sample_step)
            smp_ys = [data[x+input_size] for x in smp_xs]
            ax.scatter(smp_xs, smp_ys, label='sampling markers')
        if show_error:
            err_plt = np.array(generated_data) - np.array(data[input_size:])
            ax.plot(xs, err_plt, label='error')
            ax.plot(xs, [0]*len(xs), linestyle='--')
        plt.legend()

        if save_fig:
            assert title is not None, "Provide a title/filename to save results."
            f.savefig(title)
        plt.show()
    
    return generated_data, error

if __name__ == "__main__":
    # Experiment settings / parameters ========================================================
    t = str(time.time()).replace('.', 'p')
    eval_valid = True    # whether or not to evaluate MSE loss on test set during training
    eval_gener = True    # whether or not to generate future values, calculate that MSE loss
    eval_gen_loss = True
    save_fig = False
    save_results = False

    reg = 1e-3 # lambda for L2 regularization 
    n_generate_timesteps = 2000
    learn_rate = 0.009
    n_epochs = 100
    
    # ========================================================================================
    # Get data ===============================================================================
    from MackeyGlass.MackeyGlassGenerator import run
    data = run(num_data_samples=21000)
    data_var = np.var(np.array(data))
    __DATA_VAR__ = np.var(np.array(data))
    __DATA_MEAN__ = np.mean(np.array(data))
    print('data mean, variance: %.5f, %.5f' % (__DATA_MEAN__, __DATA_VAR__))

    train_data = data[:14000]
    if eval_valid:
        valid_data = data[14000:20000]
    else:
        valid_data = None
    test_data = data[20000:]
    
    # Set up model, loss function, optimizer =================================================
    model = FFNN(input_size=50, hidden_size=100, n_hidden_layers=2, activation=nn.Sigmoid)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    title = "%s__ninputs%d__layers%d__nHU%d__lambda%.5f__lr%f" \
            % (t, model.input_size, model.n_hidden_layers, model.hidden_size, reg, learn_rate)
    title = title.replace('.', 'p') # replace period w/ 'p' so can be used as filename
    # Train model ============================================================================
    model, stats = train(model, train_data, 20, n_epochs, criterion, optimizer, 
                         valid_data=valid_data, verbose=1, eval_gen_loss=eval_gen_loss,
                         n_generate_timesteps=n_generate_timesteps)

    # losses are NORMALIZED ROOT MEAN SQUARE ERROR (not regular MSE)
    train_losses = stats[:, 0]
    if 1:
        if eval_valid:
            valid_losses = stats[:, 1]

            f, (ax1, ax2) = plt.subplots(2, 1)
            xs = range(len(train_losses))
            ax1.plot(xs, train_losses)
            ax1.set_title('Supervised training NRMSE per epoch')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')

            xs = range(len(valid_losses))
            ax2.plot(xs, valid_losses)
            ax2.set_title('Supervised validation NRMSE per epoch')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            
            if save_fig:
                f.savefig('Results/FFNN/FIG__%s__tr-val-loss.pdf' % title)
            plt.show()
        else:
            f, ax = plt.subplots()
            xs = range(len(train_losses))
            ax.plot(xs, train_losses)
            ax.set_title('Training loss per epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
        
            if save_fig:
                f.savefig('Results/FFNN/FIG__%s__tr-loss.pdf' % title)
            
            plt.legend(); plt.show()
    
    # IMPORTANT BITS ====================================================================== 
    if eval_gener:
        input_size = model.input_size
        if valid_data is None:
            valid_data = data[14000:]

        g_title = 'Results/FFNN/FIG__%s__gen-loss.pdf' % title
        generated_outputs, gen_mse = test(
            model, valid_data[:(n_generate_timesteps+input_size)], plot=False
        )

        print('\n'*3)
        print('='*30)

        gen_mse_normed = gen_mse
        print('Final model\'s validation NRMSE for %d generated values: %.7f' % \
                (n_generate_timesteps, gen_mse_normed))


        generated_outputs_train, gen_err_train = test(
            model, train_data[:n_generate_timesteps], plot=False
        )

        print('gen_err_train: %.7f' % gen_err_train)

        import pickle as pkl
        to_save = dict()
        to_save['stats'] = stats
        to_save['model'] = model
        to_save['gen_outputs'] = generated_outputs
        to_save['gen_normed_mse'] = gen_mse_normed
        to_save['n_generated_timesteps'] = n_generate_timesteps
        to_save['adam_learn_rate'] = learn_rate

        if save_results:
            fname = 'Results/FFNN/PKL__%s.p' % title
            pkl.dump(to_save, open(fname, 'wb'))

        # PLOTTING GENERATION STUFF ========================================================
        plot_title = ''
        show_error = True
       
        input_size = model.input_size 
        xs = range(len(generated_outputs))
        f, ax = plt.subplots(figsize=(16, 10))
        
        ax.set_title(plot_title)
        ax.plot(xs, valid_data[input_size:(n_generate_timesteps+input_size)], label='True data')
        ax.plot(xs, generated_outputs, label='Generated data')
        if show_error:
            errors = np.array(generated_outputs) - \
                        np.array(valid_data[input_size:(n_generate_timesteps+input_size)])
            ax.plot(xs, errors, label='error')
            ax.plot(xs, [0]*len(xs), linestyle='--')
        plt.legend()

        if save_fig:
            f.savefig(g_title)
        plt.show()



        # PLOT GENERATION VALIDATION LOSS PER EPOCH ===================================
        f, ax = plt.subplots(figsize=(16, 10))
        ax.set_title('Generation validation NRMSE per epoch')
        xs = range(len(stats[:, 3]))
        ax.plot(xs, stats[:, 3])
        argmin_err = np.argmin(stats[:, 3])
        min_err = stats[:, 3].min()
        ax.scatter([argmin_err], [min_err], label='Minimum')
        plt.legend()
        f.savefig('Results/FFNN/FIG__Gen_val_loss.pdf')
        plt.show()



        # TEST DATA ===================================================================
        generated_outputs_test, nrmse_test = test(
            model, test_data, save_fig=True, title='Results/FFNN/FIG__test_gen.pdf'
        )

        print('TEST NRMSE: %.7f' % nrmse_test)


















pass
