from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    """ 
     input_size: Data dimensionality (i.e. MackeyGlass: 1).
    hidden_size: Number of features in each hidden state, h.
       n_layers: Number of recurrent layers.
    """

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers)
        self.linear_out = nn.Linear(hidden_size, input_size)

    def forward(self, inputs, predict_timesteps=0):
        """
        Set predict_timesteps = the number of timesteps you would like to predict/generate 
            after training on the training data 'inputs'.Your location
        """
        # inputs.size(): (seq_len, input_size)
        outputs, (h_n, c_n) = self.rnn(inputs)
        seq_len, batch_size, hidden_size = outputs.shape
        
        # reshape outputs to be put through linear layer
        outputs = outputs.view(seq_len*batch_size, hidden_size)
        outputs = self.linear_out(outputs).view(seq_len, batch_size, self.input_size)

        if not predict_timesteps:
            return outputs
        
        input_t = outputs[-1, -1, :].view(1, 1, self.input_size)
        generated_outputs = []
        for i in range(predict_timesteps):
            output_t, (h_n, c_n) = self.rnn(input_t, (h_n, c_n))
            output_t = output_t.view(1, hidden_size)
            output_t = self.linear_out(output_t).view(1, 1, self.input_size)
            generated_outputs.append(output_t.data.numpy()[0, 0, :])

            input_t = output_t
        
        generated_outputs = np.array(generated_outputs).reshape(predict_timesteps, self.input_size)

        if torch.cuda.is_available():
            return outputs, Variable(torch.FloatTensor(generated_outputs))
        else:
            return outputs, Variable(torch.FloatTensor(generated_outputs).cuda())

if __name__ == '__main__':
    from MackeyGlass.MackeyGlassGenerator import run
    data = run(12000) 
    
    train_data = np.array(data[:7000]).reshape(-1, 1, 1)
    test_data = np.array(data[7000:]).reshape(-1, 1, 1)
    # CONSTRUCT TRAINING, TESTING DATA
    if torch.cuda.is_available():
        train_inputs = Variable(torch.from_numpy(train_data[:-1]).float().cuda(), requires_grad=0)
        train_targets = Variable(torch.from_numpy(train_data[1:]).float().cuda(), requires_grad=0)
        test_inputs = Variable(torch.from_numpy(test_data[:-1]).float().cuda(), requires_grad=0)
        test_targets = Variable(torch.from_numpy(test_data[1:]).float().cuda(), requires_grad=0)
    else:
        train_inputs = Variable(torch.from_numpy(train_data[:-1]).float(), requires_grad=0)
        train_targets = Variable(torch.from_numpy(train_data[1:]).float(), requires_grad=0)
        test_inputs = Variable(torch.from_numpy(test_data[:-1]).float(), requires_grad=0)
        test_targets = Variable(torch.from_numpy(test_data[1:]).float(), requires_grad=0)

    rnn = LSTM(1, 50, n_layers=2)

    if torch.cuda.is_available():
        rnn.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)

    n_epochs = 200
    stats = np.zeros((n_epochs, 2))
    for epoch in range(n_epochs):
        print('Epoch [%d/%d] ===================' % (epoch+1, n_epochs))
        
        # calculate outputs, loss, then step
        optimizer.zero_grad()
        train_outputs = rnn(train_inputs)
        loss = criterion(train_outputs, train_targets)
        stats[epoch, 0] = loss.data.numpy()[0]
        print('Training loss: %.6f' % loss.data.numpy()[0])
        loss.backward()
        optimizer.step()

        test_outputs, generated_outputs = rnn(test_inputs, predict_timesteps=len(test_data))
        # loss = criterion(test_outputs, test_targets)
        if torch.cuda.is_available():
            loss = criterion(Variable(torch.from_numpy(test_data)), generated_outputs.double())
        else:
            loss = criterion(Variable(torch.from_numpy(test_data)).cuda(), generated_outputs.double())

        stats[epoch, 1] = loss.data.numpy()[0]
        print('Test loss: %.6f' % loss.data.numpy()[0])

    # FINAL EPOCH: try generating data as well ====================================
    print('Training finished: running generation tests now.')
    train_outputs, generated_outputs = rnn(train_inputs, predict_timesteps=len(test_data))
    if torch.cuda.is_available():
        generated_test_loss = criterion(Variable(torch.from_numpy(test_data)), generated_outputs.double())
    else:
        generated_test_loss = criterion(Variable(torch.from_numpy(test_data)).cuda(), generated_outputs.double())

    print('MSE loss for generated data: %.6f' % generated_test_loss)

    display_mode = False

    if display_mode:
        f, ax = plt.subplots(figsize=(12, 12))
        # plot true test target values
        outputs_plt = test_outputs.data.numpy().squeeze()
        targets_plt = test_targets.data.numpy().squeeze()
        xs = np.arange(len(outputs_plt))
        ax.plot(xs, targets_plt, label='True')
        ax.plot(xs, outputs_plt, label='Model')
        ax.set_title('Test outputs; true vs. predicted (no generation)')
        plt.legend(); plt.show()
    if display_mode:
        f, ax = plt.subplots(figsize=(12, 12))
        xs = np.arange(n_epochs)
        ax.plot(xs, stats[:, 0], label='Training loss')
        ax.plot(xs, stats[:, 1], label='Test loss')
        plt.legend(); plt.show()
    if display_mode:
        generated_plt = generated_outputs.data.numpy().squeeze()
        test_plt = test_data.squeeze()
        f, ax = plt.subplots(figsize=(12, 12))
        xs = np.arange(len(test_plt))
        ax.plot(xs, test_plt, label='True data')
        ax.plot(xs, generated_plt, label='Generated data')
        plt.legend(); plt.show()


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














