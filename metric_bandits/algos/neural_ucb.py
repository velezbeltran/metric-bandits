"""
The implementation of neural UCB follows very closely what is described in
the paper. Name of variables are chosen so as to agree with the paper.
`https://arxiv.org/pdf/1911.04462.pdf`
s T, regularization parameter λ, exploration parameter ν, confidence parameter δ, norm
parameter S, step size η, number of gradient descent steps J, network width m, network depth L.
"""
from math import sqrt

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metric_bandits.algos.base import BaseAlgo
from metric_bandits.utils.math import sherman_morrison


class NeuralUCB(BaseAlgo):
    def __init__(self, model, reg, step_size, num_steps, train_freq, explore_param):
        super().__init__()
        self.reg = reg
        self.step_size = step_size
        self.num_steps = num_steps
        self.explore_param = explore_param
        self.train_freq = train_freq

        # Set up model and optimizer
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=step_size)

        # state of the algorithm
        self.Z_inv = None
        self.t = 0

        # parameters to keep track of
        self.avg = 0
        self.n = 1
        self.last_action = None
        self.rewards = []
        self.contexts_played = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        self.model.eval()
        self.ucb_val_grads, self.ucb_estimate = {}, {}
        for action in actions:
            val, grad = self.get_val_grad(actions[action])
            self.ucb_val_grads[action] = (val, grad)
            opt = self.optimist_reward(grad)
            self.ucb_estimate[action] = val + opt

        # return the key with the highest value
        self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        self.contexts_played.append(actions[self.last_action])
        return self.last_action

    def update(self, reward):
        """
        Updates the model
        """
        self.rewards.append(reward)

        # update our confidence matrix
        prev_grad = self.ucb_val_grads[self.last_action][1] / sqrt(
            self.model.num_params
        )
        self.Z_inv = sherman_morrison(self.Z_inv, prev_grad.detach())

        # decide whether to train the model
        self.t += 1
        if self.t % self.train_freq == 0:
            self.train()

    def get_val_grad(self, x):
        """
        Returns the predicted value and the gradient of the neural network
        as a one dimensional column vector.
        """
        self.model.zero_grad()
        val, _ = self.model(x)
        grad = torch.autograd.grad(val, self.model.parameters(), create_graph=False)
        g = torch.cat([g.flatten() for g in grad])
        return val, g.unsqueeze(-1)

    def optimist_reward(self, grad):
        grad = grad.detach()
        with torch.no_grad():
            val = self.explore_param * torch.sqrt(
                grad.T @ self.Z_inv @ grad / self.model.num_params
            )
            return val

    def train(self):
        """
        Trains the model
        """
        self.model.train()
        inputs = torch.stack(self.contexts_played)
        tgts = torch.tensor(self.rewards).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(inputs, tgts)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.train_freq, shuffle=True
        )

        for epoch in (pbar := tqdm(range(self.num_steps))):
            for x, y in loader:
                assert x.shape[0] == self.train_freq
                assert y.shape[0] == self.train_freq
                self.optimizer.zero_grad()
                val, _ = self.model(x)
                loss = F.mse_loss(val, y, reduction="mean")
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f"Loss: {loss.item():.4f}")

        self.model.eval()
        self.save()

    def reset(self):
        """
        Resets the model
        """
        self.Z_inv = torch.eye(self.model.num_params, requires_grad=False)
        print("Reset model")
