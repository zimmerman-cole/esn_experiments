import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class FFNN(nn.Module):
    """
    Feedforward neural network for modelling (chaotic) time series data.
    
    Args:
        input_size:             number of frames of context (data for previous time steps).
        hidden_size:            number of hidden units per hidden layer.
        n_hidden_layers:        number of hidden layers.
        activation:             pytorch activation.
    """

    def __init__(self, input_size, hidden_size, n_hidden_layers, activation=None):
        super(FFNN, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(input_size)
        self.n_hidden_layers = int(n_hidden_layers)
        
        if activation is None:
            activation = nn.Sigmoid()
        self.activation = activation
        
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_size)) # input layer
        for i in range(n_hidden_layers):
            # add hidden layers
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.layers.append(nn.Linear(hidden_size, 1)) # output layer

    def forward(self, x):
        out = x    
        # forward-prop through input and hidden layers
        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = self.activation(out)

        # fprop through output layer
        out = self.layers[-1](out)
        return out


def train(model, train_data, batch_size, num_epochs, criterion, optimizer, valid_data=None):
    input_size = model.input_size
    assert (len(train_data) - input_size) % batch_size == 0, \
                "there is leftover training data that doesn't fit neatly into a batch"

    n_iter = int((len(train_data) - input_size) / batch_size)

    for epoch in range(num_epochs):
        for i in range(0, n_iter, batch_size):
            inputs = torch.FloatTensor(batch_size, input_size)
            targets = torch.FloatTensor(input_size)
            for j in range(i, i+batch_size):
                inputs[j] = torch.FloatTensor(train_data[j:(j+input_size)])
                targets[j] = train_data[j+input_size]
            inputs = Variable(inputs)
            targets = Variable(targets)

            # fprop, bprop, optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (i % 50) == 0:
                print('='*50)
                print('Epoch [%d/%d]' % (epoch+1, num_epochs))
                print('Batch [%d/%d]' % (i+1, n_iter))
                print('Training Loss: %.4f' % loss.data[0])

        if valid_data is not None:
            valid_loss = 0.
            for i in range(len(valid_data) - input_size):
                inputs = valid_data[i:(i+input_size)]
                inputs = Variable(torch.FloatTensor(inputs))
                target = Variable(torch.FloatTensor(valid_data[i+input_size]))

                output = model(inputs)
                valid_loss += criterion(output, target)

            if criterion.size_average:
                valid_loss /= (len(valid_data) - input_size)
                print('Validation MSE: %.4f' % valid_loss)
            else:
                print('Validation total SE: %.4f' % valid_loss)
                

    return model


if __name__ == "__main__":
    from MackeyGlass.MackeyGlassGenerator import run
    
    data = run()
    
    model = FFNN(19, 50, 5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    

    #train(model, 



