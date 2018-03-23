from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np


# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()


# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)


# kwargs = {'num_workers': 1, 'pin_memory': False}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=32, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=32, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size=None, latent_variable_size=5,
                    log_interval=2, batch_size=32, epochs=3):
        super(VAE, self).__init__()

        self.input_size = input_size
        if hidden_size is None: self.hidden_size = (input_size + latent_variable_size) / 2
        else: self.hidden_size = hidden_size
        self.latent_variable_size = latent_variable_size

        self.log_interval = log_interval
        self.batch_size = batch_size
        self.epochs = epochs

        # self.fc1 = nn.Linear(784, 400)
        # self.fc21 = nn.Linear(400, 20)
        # self.fc22 = nn.Linear(400, 20)
        # self.fc3 = nn.Linear(20, 400)
        # self.fc4 = nn.Linear(400, 784)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, self.latent_variable_size)
        self.fc22 = nn.Linear(self.hidden_size, self.latent_variable_size)
        self.fc3 = nn.Linear(self.latent_variable_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        self.avg_loss_history = []

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        # return self.sigmoid(self.fc4(h3))
        return self.fc4(h3) # linear ouput for predicting dynamical data

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size), size_average=False)
        BCE = F.mse_loss(recon_x, x.view(-1, self.input_size), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train_one_epoch(self, epoch_idx, train_data, verbose=False):
        self.train()
        train_loss = 0
        # for batch_idx, data in enumerate(train_data):
        for batch_idx in range(0, len(train_data), self.batch_size):
            # data = Variable(data)
            data = Variable(train_data[batch_idx:(batch_idx+self.batch_size)])
            # if args.cuda:
            #     data = data.cuda()
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            self.optimizer.step()
            if batch_idx % self.log_interval == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx * len(data), len(train_data), #len(self.train_data.dataset),
                    100. * batch_idx / len(train_data),
                    loss.data[0] / len(data)))

        # print('====> Epoch: {} Average loss: {:.8f}'.format(
        #     epoch_idx, train_loss / len(train_data)))
            # epoch, train_loss / len(self.train_data.dataset)))
        self.avg_loss_history.append(train_loss / len(train_data))

    def test(self, epoch_idx, test_data):
        self.eval()
        test_loss = 0
        for i, data in enumerate(test_data):
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self(data)
            test_loss += self.loss_function(recon_batch, data, mu, logvar).data[0]
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                         recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.data,
            #             'results/reconstruction_' + str(epoch_idx) + '.png', nrow=n)

        # test_loss /= len(self.test_data.dataset)
        test_loss /= len(test_data)
        # print('====> {} Test set loss: {:.4f}'.format(epoch_idx, test_loss))
    
    def train_full(self, train_data, test_data=None, verbose=False):
        # normalise the data before putting it into the ENCODER
        # _mean = train_data.mean()
        # _std = train_data.std()
        # train_data -= _mean
        # train_data /= _std.mean()
        # _std = train_data.std()
        # train_data -= _mean
        # train_data /= _std

        for epoch in range(1, self.epochs + 1):
            train_shuff = train_data[torch.randperm(train_data.shape[0]), :]
            # print(train_data[:5, :5])
            # print(train_shuff[:5, :5])

            self.train_one_epoch(epoch, train_shuff)
            if test_data is not None:
                self.test(epoch, test_data)
            self.sample()

    def sample(self):
        # print(torch.sort(torch.randn(200, self.latent_variable_size), dim=0))
        sample = Variable(torch.sort(torch.randn(200, self.latent_variable_size), dim=0)[0])
        sample = self.decode(sample)
        # print(sample)
        # print(sample.size())
        # save_image(sample.data.view(64, 1, 15, 20)*255,
        #         'sample_' + str(epoch) + '.png')
        return sample.data.numpy()

# model = VAE()
# if args.cuda:
#     model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD

# # Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return BCE + KLD


# def train(epoch, model, optimiser):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = Variable(data)
#         # if args.cuda:
#         #     data = data.cuda()
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.data[0]
#         optimizer.step()
#         if batch_idx % model.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data[0] / len(data)))

#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))


# def test(epoch, model):
#     model.eval()
#     test_loss = 0
#     for i, (data, _) in enumerate(test_loader):
#         data = Variable(data, volatile=True)
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
#         if i == 0:
#             n = min(data.size(0), 8)
#             comparison = torch.cat([data[:n],
#                                   recon_batch.view(model.batch_size, 1, 28, 28)[:n]])
#             save_image(comparison.data,
#                      'results/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

# model = VAE(784)
# model.train_full()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# for epoch in range(1, model.epochs + 1):
#     train(epoch, model, optimizer)
#     test(epoch, model)
#     sample = Variable(torch.randn(64, 20))
#     sample = model.decode(sample)
#     save_image(sample.data.view(64, 1, 28, 28),
#                'results/sample_' + str(epoch) + '.png')
