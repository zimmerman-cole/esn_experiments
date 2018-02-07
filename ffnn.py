from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

import matplotlib.pyplot as plt


class FFNN(nn.Module):
    """
    Feedforward neural network for modelling (chaotic) time series data.
    
    Args:
        input_size:             number of frames of context (data for previous time steps).
        hidden_size:            number of hidden units per hidden layer.
        n_hidden_layers:        number of hidden layers (not including input+output layers).
        activation:             pytorch activation (class, NOT an instance)
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

def train(model, train_data, batch_size, num_epochs, criterion, optimizer, valid_data=None, verbose=1):
    input_size = model.input_size
    #assert (len(train_data) - input_size) % batch_size == 0, \
    #            "there is leftover training data that doesn't fit neatly into a batch"

    n_iter = int((len(train_data) - input_size) / batch_size)

    if valid_data is not None:
        stats = torch.FloatTensor(num_epochs, 2)
    else:
        stats = torch.FloatTensor(num_epochs, 1)

    for epoch in range(num_epochs):
        train_loss = 0.
        for i in range(0, n_iter, batch_size):
            inputs = torch.FloatTensor(batch_size, input_size)
            targets = torch.FloatTensor(batch_size)
            for batch_idx, j in enumerate(range(i, i+batch_size)):
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

            train_loss += loss.data[0]

        if criterion.size_average:
            train_loss /= float(n_iter)

        stats[epoch, 0] = train_loss
        if verbose:
            print('='*50)
            print('Epoch [%d/%d]' % (epoch+1, num_epochs))
            print('Total training loss: %.7f' % train_loss)

        if valid_data is not None:
            valid_loss = 0.
            for i in range(len(valid_data) - input_size):
                inputs = valid_data[i:(i+input_size)]
                inputs = Variable(torch.FloatTensor(inputs))
                target = Variable(torch.FloatTensor([valid_data[i+input_size]]))

                output = model(inputs)
                valid_loss += criterion(output, target).data[0]

            if criterion.size_average:
                valid_loss /= (len(valid_data) - input_size)
            
            stats[epoch, 1] = valid_loss
            if verbose:
                print('Total validation loss: %.7f' % valid_loss)

    return model, stats

def test(model, data):
    """ Pass the trained model. """
    input_size = model.input_size

    inputs = data[:input_size] # type(inputs) = list
    output = model(Variable(torch.FloatTensor(inputs))).data[0]
    generated_data = [output]
    
    for i in range(len(data) - input_size - 1):
        inputs.extend([output]) # shift input
        inputs = inputs[1:]     # data

        output = model(Variable(torch.FloatTensor(inputs))).data[0]
        generated_data.append(output)

    xs = range(len(data) - input_size)
    f, ax = plt.subplots()
    #print(len(xs), len(data[input_size:]), len(generated_data))
    ax.plot(xs, data[input_size:], label='True data')
    ax.plot(xs, generated_data, label='Generated data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from MackeyGlass.MackeyGlassGenerator import run 
    data = run(num_data_samples=12000)
    train_data = data[:7000]; valid_data = data[7000:]
    model = FFNN(input_size=20, hidden_size=100, n_hidden_layers=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model, stats = train(model, train_data, 20, 100, criterion, optimizer, valid_data=None, verbose=1)
    train_losses = stats[:, 0].numpy()
    #valid_losses = stats[:, 1].numpy()

    if 0:
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
        plt.show()

    if 1:
        test(model, valid_data[:100])

