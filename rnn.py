from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl
from copy import deepcopy

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
            generated_outputs.append(output_t.data.cpu().numpy()[0, 0, :])

            input_t = output_t
            # print(h_n, c_n)

        generated_outputs = np.array(generated_outputs).reshape(predict_timesteps, self.input_size)

        if torch.cuda.is_available():
            return outputs, Variable(torch.FloatTensor(generated_outputs).cuda())
        else:
            return outputs, Variable(torch.FloatTensor(generated_outputs))

def nrmse(y_pred, y_target, DATA_MEAN):
	return np.sqrt(np.sum((y_pred - y_target)**2)/np.sum((y_target - DATA_MEAN)**2))

# if __name__ == '__main__':

#     from MackeyGlass.MackeyGlassGenerator import run
#     data = run(21000)
#     data -= np.mean(data)
#     DATA_MEAN = np.mean(data)

#     train_data = np.array(data[:1000]).reshape(-1, 1, 1)
#     test_data = np.array(data[1000:1500]).reshape(-1, 1, 1)

#     # CONSTRUCT TRAINING, TESTING DATA
#     if torch.cuda.is_available():
#         print('CUDA available!')
#         train_inputs = Variable(torch.from_numpy(train_data[:-1]).float().cuda(), requires_grad=0)
#         train_targets = Variable(torch.from_numpy(train_data[1:]).float().cuda(), requires_grad=0)
#         test_inputs = Variable(torch.from_numpy(test_data[:-1]).float().cuda(), requires_grad=0)
#         test_targets = Variable(torch.from_numpy(test_data[1:]).float().cuda(), requires_grad=0)
#     else:
#         train_inputs = Variable(torch.from_numpy(train_data[:-1]).float(), requires_grad=0)
#         train_targets = Variable(torch.from_numpy(train_data[1:]).float(), requires_grad=0)
#         test_inputs = Variable(torch.from_numpy(test_data[:-1]).float(), requires_grad=0)
#         test_targets = Variable(torch.from_numpy(test_data[1:]).float(), requires_grad=0)

#     rnn = LSTM(1, 100, n_layers=1)

#     if torch.cuda.is_available():
#         print("RUNNING CUDA!")
#         rnn.cuda()
#     else:
#         print("NOT RUNNING CUDA!")

#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(rnn.parameters(), lr=0.001)

#     n_epochs = 20
#     stats = np.zeros((n_epochs, 2))
#     for epoch in range(n_epochs):
#         print('Epoch [%d/%d] ===================' % (epoch+1, n_epochs))

#         # calculate outputs, loss, then step
#         optimizer.zero_grad()
#         train_outputs = rnn(train_inputs, predict_timesteps=0)
#         loss = criterion(train_outputs, train_targets)
#         #stats[epoch, 0] = loss.data.cpu().numpy()[0]
#         #print('Training MSE loss: %.6f' % stats[epoch, 0])
#         loss.backward()
#         optimizer.step()

#         #test_outputs, generated_outputs = rnn(test_inputs, predict_timesteps=len(test_data))
#         # loss = criterion(test_outputs, test_targets)
#         #if torch.cuda.is_available():
#             #loss = criterion(Variable(torch.from_numpy(test_data)), generated_outputs.double())
#         #else:
#         train_outputs, train_gen = rnn(train_inputs[:100], predict_timesteps=len(train_inputs)-100)
#         # print(train_outputs.data.numpy())
#         # print(train_gen.data.numpy())
#         # test_outputs, test_gen = rnn(test_inputs[:100], predict_timesteps=len(test_inputs)-100)
#             #loss = criterion(Variable(torch.from_numpy(test_data)).cuda(), generated_outputs.double())
#         if epoch % 1 == 0:
#             if torch.cuda.is_available():
#                 nrmse_sup_train = nrmse(train_outputs.cpu().data.numpy(), train_targets.cpu().data.numpy(), DATA_MEAN)
#                 nrmse_sup_test = nrmse(test_outputs.cpu().data.numpy(), test_targets.cpu().data.numpy(), DATA_MEAN)
#                 nrmse_gen_train = nrmse(train_gen.cpu().data.numpy(), train_targets.cpu().data.numpy(), DATA_MEAN)
#                 nrmse_gen_test = nrmse(test_gen.cpu().data.numpy(), test_targets.cpu().data.numpy(), DATA_MEAN)
#             else:
#                 # nrmse_sup_train = nrmse(train_outputs.data.numpy(), train_targets.data.numpy(), DATA_MEAN)
#                 # nrmse_sup_test = nrmse(test_outputs.data.numpy(), test_targets.data.numpy(), DATA_MEAN)
#                 nrmse_gen_train = nrmse(train_gen.data.numpy(), train_targets[100:].data.numpy(), DATA_MEAN)
#                 # nrmse_gen_test = nrmse(test_gen.data.numpy(), test_targets[100:].data.numpy(), DATA_MEAN)

#         #stats[epoch, 1] = loss.data.cpu().numpy()[0]
#         #print('Test loss: %.6f' % stats[epoch, 1])
#         # stats[epoch, 0] = nrmse_sup_train
#         # stats[epoch, 1] = nrmse_sup_test
#         # print('RMSE (SUP) -- TRAIN: {}, TEST: {}'.format(nrmse_sup_train, nrmse_sup_test))
#         # print('RMSE (GEN) -- TRAIN: {}, TEST: {}'.format(nrmse_gen_train, nrmse_gen_test))
#         print('RMSE (GEN) -- TRAIN: {}'.format(nrmse_gen_train))

#     # FINAL EPOCH: try generating data as well ====================================
#     print('Training finished: running generation tests now.')
#     train_outputs, generated_outputs = rnn(train_inputs, predict_timesteps=len(test_data))
#     if torch.cuda.is_available():
#         generated_test_loss = criterion(Variable(torch.from_numpy(test_data)).cuda(), generated_outputs.double())
#     else:
#         generated_test_loss = criterion(Variable(torch.from_numpy(test_data)), generated_outputs.double())

#     print('MSE loss for generated data: %.6f' % generated_test_loss)

#     display_mode = True

#     if display_mode:    
#         f, ax = plt.subplots(figsize=(12, 12))
#         # plot true test target values
#         outputs_plt = train_outputs.data.cpu().numpy().squeeze()
#         targets_plt = train_targets.data.cpu().numpy().squeeze()
#         #outputs_plt = test_outputs.data.cpu().numpy().squeeze()
#         #targets_plt = test_targets.data.cpu().numpy().squeeze()
#         xs = np.arange(len(outputs_plt))
#         ax.plot(xs, targets_plt, label='True')
#         ax.plot(xs, outputs_plt, label='Model')
#         ax.set_title('Test outputs; true vs. predicted (no generation)')
#         plt.legend(); plt.show()
#     if display_mode and 0:
#         f, ax = plt.subplots(figsize=(12, 12))
#         # plot true test target values
#         outputs_plt = test_outputs.data.cpu().numpy().squeeze()
#         targets_plt = test_targets.data.cpu().numpy().squeeze()
#         #outputs_plt = test_outputs.data.cpu().numpy().squeeze()
#         #targets_plt = test_targets.data.cpu().numpy().squeeze()
#         xs = np.arange(len(outputs_plt))
#         ax.plot(xs, targets_plt, label='True')
#         ax.plot(xs, outputs_plt, label='Model')
#         ax.set_title('Test outputs; true vs. predicted (no generation)')
#         plt.legend(); plt.show()
#     if display_mode and 0:
#         f, ax = plt.subplots(figsize=(12, 12))
#         xs = np.arange(n_epochs)
#         ax.plot(xs, stats[:, 0], label='Training loss')
#         ax.plot(xs, stats[:, 1], label='Test loss')
#         plt.legend(); plt.show()
#     if display_mode:
#         train_outputs, train_gen = rnn(train_inputs[:100], predict_timesteps=len(train_inputs)-100)
#         generated_plt = train_gen.data.cpu().numpy().squeeze()
#         test_plt = train_inputs[100:].cpu().data.numpy().squeeze()
#         f, ax = plt.subplots(figsize=(12, 12))
#         xs = np.arange(len(test_plt))
#         ax.plot(xs, test_plt, label='True data')
#         ax.plot(xs, generated_plt, label='Generated data')
#         plt.legend(); plt.show()
#     if display_mode and 0:
#         generated_plt = generated_outputs.data.cpu().numpy().squeeze()
#         test_plt = test_data.squeeze()
#         f, ax = plt.subplots(figsize=(12, 12))
#         xs = np.arange(len(test_plt))
#         ax.plot(xs, test_plt, label='True data')
#         ax.plot(xs, generated_plt, label='Generated data')
#         plt.legend(); plt.show()


class Sequence(nn.Module):

    def __init__(self):
        super(Sequence, self).__init__()
        self.hidden_size = 50
        self.num_hid_layers = 1
        self.layers = OrderedDict()
        self.layers['sigmoid'] = nn.Linear(1, self.hidden_size)
        # self.layers['lstm0'] = nn.LSTMCell(1, self.hidden_size)
        for i in range(self.num_hid_layers):
            self.layers['lstm{}'.format(i+1)] = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.layers['linear'] = nn.Linear(self.hidden_size, 1)
        # self.lstm1 = nn.LSTMCell(1, self.hidden_size)
        # self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        # self.gru1 = nn.GRUCell(1, self.hidden_size)
        # self.gru2 = nn.GRUCell(self.hidden_size, self.hidden_size)
        # self.linear = nn.Linear(self.hidden_size, 1)
        self.model = nn.Sequential(self.layers)

    def forward(self, inputs, future=0):
        outputs = []

        hids = []
        states = []
        for _ in range(self.num_hid_layers):
            hids.append(Variable(torch.zeros(1, self.hidden_size).double(), requires_grad=False))
            states.append(Variable(torch.zeros(1, self.hidden_size).double(), requires_grad=False))
            
        # h_t = Variable(torch.zeros(1, self.hidden_size).double(), requires_grad=False)
        # c_t = Variable(torch.zeros(1, self.hidden_size).double(), requires_grad=False)
        # h_t2 = Variable(torch.zeros(1, self.hidden_size).double(), requires_grad=False)
        # c_t2 = Variable(torch.zeros(1, self.hidden_size).double(), requires_grad=False)

        # for i, input_t in enumerate(inputs.chunk(inputs.size(1), dim=1)):
        # print(inputs.size())
        for i in range(inputs.size(0)):
            input_t = inputs[i, :]
            # print(input_t.size())
            input_t = F.sigmoid(self.layers['sigmoid'](input_t))
            h = 0
            val = input_t
            for k in self.layers:
                if k != 'linear' and k != 'sigmoid':
                    hids[h], states[h] = self.layers[k](val, (hids[h], states[h]))
                    val = hids[h]
                    h += 1
            # h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            # h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # h_t = self.gru1(input_t, h_t)
            # h_t2 = self.gru2(h_t, h_t2)
            # print(h_t2.size())
            output = self.layers['linear'](hids[-1])
            outputs += [output]

        # print(output)
        for i in range(future):
            # h_t, c_t = self.lstm1(output, (h_t, c_t))
            # h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = F.sigmoid(self.layers['sigmoid'](output))
            h = 0
            val = output
            for k in self.layers:
                if k != 'linear' and k != 'sigmoid':
                    hids[h], states[h] = self.layers[k](val, (hids[h], states[h]))
                    val = hids[h]
                    h += 1
            # h_t = self.gru1(input_t, h_t)
            # h_t2 = self.gru2(h_t, h_t2)
            # output = self.linear(h_t2)
            output = self.layers['linear'](hids[-1])
            outputs += [output]

        outputs = torch.stack(outputs, 0).squeeze(2)
        return outputs

def load_and_run_model(file_name):
    model = pkl.load(open(file_name, "rb"))
    (seq, gen_train_loss, gen_test_loss, epoch) = model
    print("MODEL LOADED")

    from MackeyGlass.MackeyGlassGenerator import run
    data = run(21000)
    data -= np.mean(data)
    DATA_MEAN = np.mean(data)
    print("DATA LOADED")

    train_data = np.array(data[:14000])
    train_inputs = Variable(torch.from_numpy(train_data.reshape(-1, 1)), requires_grad=0)
    test_data = np.array(data[14000:20000])
    test_targets = Variable(torch.from_numpy(test_data.reshape(-1, 1)), requires_grad=0)
    val_data = np.array(data[20000:])
    val_targets = Variable(torch.from_numpy(val_data.reshape(-1, 1)), requires_grad=0)

    pre_val_inputs = Variable(torch.from_numpy(np.array(data[:20000]).reshape(-1, 1)), requires_grad=0)

    gen_train_outs = seq.forward(train_inputs[:4000], future=2000).data.numpy()
    gen_train_loss = nrmse(gen_train_outs[4000:], train_inputs[4000:6000].data.numpy(), DATA_MEAN)
    print('!! Gen Train loss: {}'.format(gen_train_loss))
    
    gen_test_outs = seq.forward(train_inputs, future=2000).data.numpy()
    gen_test_loss = nrmse(gen_test_outs[14000:], test_targets.data.numpy()[:2000], DATA_MEAN)
    print('!! Gen Test loss: {}'.format(gen_test_loss))

    gen_val_outs = seq.forward(pre_val_inputs, future=1000).data.numpy()
    gen_val_loss = nrmse(gen_val_outs[20000:], val_targets.data.numpy()[:1000], DATA_MEAN)
    print('!! Gen Val loss: {}'.format(gen_val_loss))

    plt.plot(range(len(data)), data, label="T")
    plt.plot(range(len(gen_val_outs)), gen_val_outs,  label="G")
    plt.legend()
    plt.show()

def try_toy_example():
    from MackeyGlass.MackeyGlassGenerator import run
    data = run(20000)
    data -= np.mean(data)
    DATA_MEAN = np.mean(data)
    train_data = np.array(data[:14000]); test_data = np.array(data[14000:])
    # CONSTRUCT TRAINING, TESTING DATA
    train_inputs = Variable(torch.from_numpy(train_data[:-1].reshape(-1, 1)), requires_grad=0)
    train_targets = Variable(torch.from_numpy(train_data[1:].reshape(-1, 1)), requires_grad=0)
    test_inputs = Variable(torch.from_numpy(test_data[:-1].reshape(-1, 1)), requires_grad=0)
    test_targets = Variable(torch.from_numpy(test_data[1:].reshape(-1, 1)), requires_grad=0)
    # print(train_inputs)
    seq = Sequence()
    seq.double() # ??? what does this do?

    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=0.01)

    bestGenTrain = 1000
    bestGenTest = 1000
    epoch = -1
    bestModel = (None, bestGenTrain, bestGenTest, epoch)

    NUM_EPOCH = 500
    for i in range(NUM_EPOCH):
        print('Epoch [{}/{}]'.format(i+1, NUM_EPOCH))

        # calculate outputs, loss, then step
        optimizer.zero_grad()
        train_outputs = seq(train_inputs)
        loss = criterion(train_outputs, train_targets)
        print('Training loss: %.6f' % loss.data.cpu().numpy()[0])
        loss.backward()
        optimizer.step()

        test_outputs = seq(test_inputs, future=0)
        loss = criterion(test_outputs, test_targets)
        print('Test loss: %.6f' % loss.data.cpu().numpy()[0])
        
        if i % 5 == 0:
            gen_outs = seq.forward(train_inputs[:4000], future=2000).data.numpy()
            gen_train_loss = nrmse(gen_outs[4000:], train_targets.data.numpy()[4000:6000], DATA_MEAN)
            print('!! Gen Train loss: {}'.format(gen_train_loss))

            gen_outs = seq.forward(train_targets[-4000:], future=2000).data.numpy()
            gen_test_loss = nrmse(gen_outs[4000:], test_inputs.data.numpy()[:2000], DATA_MEAN)
            print('!! Gen Test loss: {}'.format(gen_test_loss))

            if gen_test_loss <= bestModel[2]:
                bestModel = (deepcopy(seq), gen_train_loss, gen_test_loss, i)
                pkl.dump(bestModel, open("bestModel_50hids.pkl", "wb"))
                print("---BEST SAVED")

    f, ax = plt.subplots(figsize=(12, 12))
    # plot true test target values
    out_plt = test_outputs.data.cpu().numpy(); tar_plt = test_targets.data.cpu().numpy()
    ax.plot(np.arange(len(out_plt)), tar_plt, label='True')
    ax.plot(np.arange(len(out_plt)), out_plt, label='Generated')
    
    # generate data for final model
    f2, ax2 = plt.subplots(figsize=(12, 12))
    outs = seq.forward(train_inputs[:100], future=2000).data.numpy()
    ax2.plot(np.arange(len(train_targets.data.numpy()[100:2100])), train_targets.data.numpy()[100:2100], label="True")
    ax2.plot(np.arange(len(outs[100:])), outs[100:2100], label="Predicted")
    ax2.set_title("GENERATIVE TRAIN DATA (FINAL MODEL)")
    print("FINAL GEN. LOSS FINAL MODEL --- TRAIN {}".format(nrmse(outs[100:], train_targets.data.numpy()[100:2100], DATA_MEAN)))

    f2, ax2 = plt.subplots(figsize=(12, 12))
    outs = seq.forward(test_inputs[:100], future=2000).data.numpy()
    ax2.plot(np.arange(len(test_targets.data.numpy()[100:2100])), test_targets.data.numpy()[100:2100], label="True")
    ax2.plot(np.arange(len(outs[100:])), outs[100:2100], label="Predicted")
    ax2.set_title("GENERATIVE TEST DATA (FINAL MODEL)")
    print("FINAL GEN. LOSS FINAL MODEL --- TRAIN {}".format(nrmse(outs[100:], test_targets.data.numpy()[100:2100], DATA_MEAN)))

    # generate data for best model
    f2, ax2 = plt.subplots(figsize=(12, 12))
    outs = bestModel[0].forward(train_inputs[:100], future=2000).data.numpy()
    ax2.plot(np.arange(len(train_targets.data.numpy()[100:2100])), train_targets.data.numpy()[100:2100], label="True")
    ax2.plot(np.arange(len(outs[100:])), outs[100:2100], label="Predicted")
    ax2.set_title("GENERATIVE TRAIN DATA (BEST MODEL)")
    print("FINAL GEN. LOSS BEST MODEL --- TRAIN {}".format(nrmse(outs[100:], train_targets.data.numpy()[100:2100], DATA_MEAN)))

    f2, ax2 = plt.subplots(figsize=(12, 12))
    outs = bestModel[0].forward(test_inputs[:100], future=2000).data.numpy()
    ax2.plot(np.arange(len(test_targets.data.numpy()[100:2100])), test_targets.data.numpy()[100:2100], label="True")
    ax2.plot(np.arange(len(outs[100:])), outs[100:2100], label="Predicted")
    ax2.set_title("GENERATIVE TEST DATA (BEST MODEL)")
    print("FINAL GEN. LOSS BEST MODEL --- TEST {}".format(nrmse(outs[100:], test_targets.data.numpy()[100:2100], DATA_MEAN)))

    plt.legend(); plt.show()

if __name__ == '__main__':
    try_toy_example()
    #load_and_run_model("bestModel.pkl")












