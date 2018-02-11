from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

class LSTMNN(nn.Module):
    """ doc """

    def __init__(self, input_size, hidden_size, n_rec_layers=1, activation=None):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_rec_layers
        if activation is None:
            activation == nn.Sigmoid

        self.layers = OrderedDict()
        self.layers['linear_in'] = nn.Linear(input_size, hidden_size)
        self.layers['activ_in'] = activation()
        for i in range(n_rec_layers):
            self.layers['lstm%d' % (i+1)] = nn.LSTMCell(hidden_size, hidden_size)

        self.layers['linear_out'] = nn.Linear(hidden_size, input_size)
        

    def forward(self, inputs):
        # inputs.size(): (batch_size, input_size)
        outputs = []
        



class Sequence(nn.Module):
    
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, inputs, future=0):
        outputs = []
        h_t = Variable(torch.zeros(inputs.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(inputs.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(inputs.size(0), 51).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(inputs.size(0), 51).double(), requires_grad=False)

        for i, input_t in enumerate(inputs.chunk(inputs.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

def try_toy_example():
    from MackeyGlass.MackeyGlassGenerator import run
    data = run(1000)
    train_data = np.array(data[:500]); test_data = np.array(data[500:])
    # CONSTRUCT TRAINING, TESTING DATA
    train_inputs = Variable(torch.from_numpy(train_data[:-1].reshape(-1, 1)), requires_grad=0)
    train_targets = Variable(torch.from_numpy(train_data[1:].reshape(-1, 1)), requires_grad=0)
    test_inputs = Variable(torch.from_numpy(test_data[:-1].reshape(-1, 1)), requires_grad=0)
    test_targets = Variable(torch.from_numpy(test_data[1:].reshape(-1, 1)), requires_grad=0)

    seq = Sequence()
    seq.double() # ??? what does this do?

    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=0.005)

    for i in range(300):
        print('Epoch [%d/100]' % (i+1))
        
        # calculate outputs, loss, then step
        optimizer.zero_grad()
        train_outputs = seq(train_inputs)
        loss = criterion(train_outputs, train_targets)
        print('Training loss: %.6f' % loss.data.numpy()[0])
        loss.backward()
        optimizer.step()

        test_outputs = seq(test_inputs, future=0)
        loss = criterion(test_outputs, test_targets)
        print('Test loss: %.6f' % loss.data.numpy()[0])
        
    f, ax = plt.subplots(figsize=(12, 12))
    # plot true test target values
    out_plt = test_outputs.data.numpy(); tar_plt = test_targets.data.numpy()
    ax.plot(np.arange(len(out_plt)), tar_plt, label='True')
    ax.plot(np.arange(len(out_plt)), out_plt, label='Generated')
    plt.legend(); plt.show()


if __name__ == '__main__':
    try_toy_example()












