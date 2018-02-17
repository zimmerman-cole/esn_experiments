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
    stats = torch.zeros(num_epochs, 4)

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
            nrmse = np.sqrt(loss.data[0] / __DATA_VAR__)
            train_loss += nrmse

        if criterion.size_average:
            train_loss /= float(n_iter)

        stats[epoch, 0] = train_loss
        if verbose:
            print('='*50)
            print('Epoch [%d/%d]' % (epoch+1, num_epochs))
            print('Total sup. training NRMSE: %.7f' % train_loss)

        # Calculate GENERATION training loss ======================================
        if eval_gen_loss:
            gen_outs, nrmse = test(model, train_data[:n_generate_timesteps], plot=False)
            print('Generation training NRMSE (for %d time steps): %.7f' % \
                    (n_generate_timesteps, nrmse))
            stats[epoch, 2] = nrmse

        if valid_data is not None:
            # Calculate SUPERVISED validation loss ================================
            valid_loss = 0.
            for i in range(len(valid_data) - input_size):
                inputs = valid_data[i:(i+input_size)]
                inputs = Variable(torch.FloatTensor(inputs))
                target = Variable(torch.FloatTensor([valid_data[i+input_size]]))

                output = model(inputs)
                mse = criterion(output, target).data[0]
                nrmse = np.sqrt(mse / __DATA_VAR__)
                valid_loss += nrmse

            if criterion.size_average:
                valid_loss /= (len(valid_data) - input_size)

            stats[epoch, 1] = valid_loss
            if verbose:
                print('Total sup. validation NRMSE: %.7f' % valid_loss)

            if eval_gen_loss:
                # Now calculate GENERATION validation loss ===========================
                gen_outs, nrmse = test(model, valid_data[:n_generate_timesteps], plot=False)
                print('Generation validation NRMSE (for %d time steps): %.7f' % \
                        (n_generate_timesteps, nrmse))
                stats[epoch, 3] = nrmse
                
                if nrmse <= best_model[1]:
                    best_model = (deepcopy(model), nrmse, epoch)
    
    if eval_gen_loss:
        return best_model[0], stats
    else:
        return model, stats

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
    
    # MSE
    error = np.mean((np.array(generated_data) - np.array(data[input_size:]))**2)
    # normalized RMSE
    error = np.sqrt(error / __DATA_VAR__)

    # print('MSE: %.7f' % error)
    if plot:
        xs = range(len(data) - input_size)
        f, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.plot(xs, data[input_size:], label='True data')
        ax.plot(xs, generated_data, label='Generated data')
        if sample_step is not None:
            smp_xs = np.arange(0, len(xs), sample_step)
            smp_ys = [data[x+input_size] for x in smp_xs]
            ax.scatter(smp_xs, smp_ys, label='sampling markers')
        if show_error:
            ax.plot(xs, error, label='error')
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
    eval_valid = True # whether or not to evaluate MSE loss on test set during training
    eval_gener = True # whether or not to generate future values, calculate that MSE loss
    save_fig = True

    reg = 0. # lambda for L2 regularization 

    n_generate_timesteps = 2000
    learn_rate = 0.0001

    n_epochs = 500
    # ========================================================================================
    # Get data ===============================================================================
    from MackeyGlass.MackeyGlassGenerator import run
    data = run(num_data_samples=20000)
    data_var = np.var(np.array(data))
    __DATA_VAR__ = np.var(np.array(data))

    train_data = data[:14000]
    if eval_valid:
        valid_data = data[14000:]
    else:
        valid_data = None
    
    # Set up model, loss function, optimizer =================================================
    model = FFNN(input_size=50, hidden_size=100, n_hidden_layers=2, activation=nn.Sigmoid)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    title = "%s__ninputs%d__layers%d__nHU%d__lambda%.5f" \
            % (t, model.input_size, model.n_hidden_layers, model.hidden_size, reg)
    title = title.replace('.', 'p') # replace period w/ 'p' so can be used as filename
    # Train model ============================================================================
    model, stats = train(model, train_data, 20, n_epochs, criterion, optimizer, 
                         valid_data=valid_data, verbose=1, eval_gen_loss=True,
                         n_generate_timesteps=n_generate_timesteps)

    # losses are NORMALIZED ROOT MEAN SQUARE ERROR (not regular MSE)
    train_losses = stats[:, 0].numpy()
    if eval_valid:
        valid_losses = stats[:, 1].numpy()

        f, (ax1, ax2) = plt.subplots(2, 1)
        xs = range(len(train_losses))
        ax1.plot(xs, train_losses)
        ax1.set_title('Training loss per epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        xs = range(len(valid_losses))
        ax2.plot(xs, valid_losses)
        ax2.set_title('Validation loss per epoch')
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

    if eval_gener:
        g_title = 'Results/FFNN/FIG__%s__gen-loss.pdf' % title
        generated_outputs, gen_mse = test(
            model, valid_data[:n_generate_timesteps], sample_step=None, show_error=0, \
            save_fig=save_fig, title=g_title
        )

        gen_mse_normed = gen_mse
        print('Best validation NRMSE for %d generated values: %.7f' % \
                (n_generate_timesteps, gen_mse_normed))

        import pickle as pkl
        to_save = dict()
        to_save['stats'] = stats
        to_save['model'] = model
        to_save['gen_outputs'] = generated_outputs
        to_save['gen_normed_mse'] = gen_mse_normed
        to_save['n_generated_timesteps'] = n_generate_timesteps
        to_save['adam_learn_rate'] = learn_rate

        fname = 'Results/FFNN/PKL__%s.p' % title
        pkl.dump(to_save, open(fname, 'wb'))


