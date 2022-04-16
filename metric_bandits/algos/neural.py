"""
The implementation of neural UCB follows very closely what is described in
the paper. Name of variables are chosen so as to agree with the paper.
`https://arxiv.org/pdf/1911.04462.pdf`
s T, regularization parameter λ, exploration parameter ν, confidence parameter δ, norm
parameter S, step size η, number of gradient descent steps J, network width m, network depth L.
"""
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metric_bandits.algos.base import BaseAlgo
from metric_bandits.utils.nn import make_metric


class Neural(BaseAlgo):
    def __init__(
        self,
        model,
        step_size,
        num_steps,
        train_freq,
        explore_param,
        verbose=True,
    ):
        """
        If active is true, the model forgets completely about regret and just takes actions
        with the aim of maximizing information gain
        """
        super().__init__()
        self.step_size = step_size
        self.num_steps = num_steps
        self.explore_param = explore_param
        self.train_freq = train_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # Set up model and optimizer
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=step_size)

        # state of the algorithm
        self.t = 0
        self.train_t = 0

        # parameters to keep track of
        self.last_action = None
        self.rewards = []
        self.contexts_played = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        greedy = random.random() < self.explore_param
        if greedy:
            self.vals = {}
            for action in actions:
                val = self.model(actions[action])
                self.vals[action] = val.item()
            self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        else:
            self.last_action = random.choice(list(actions.keys()))
        self.contexts_played.append(actions[self.last_action])
        return self.last_action

    def update(self, reward):
        """
        Updates the model
        """
        self.rewards.append(reward)

        # decide whether to train the model
        self.t += 1
        if self.t % self.train_freq == 0:
            self.train()

    def train(self):
        """
        Trains the model
        """
        self.model.train()
        inputs = torch.stack(self.contexts_played)
        tgts = torch.tensor(self.rewards, device=self.device).unsqueeze(-1)
        dataset = torch.utils.data.TensorDataset(inputs, tgts)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        pbar = tqdm(total=self.num_steps, disable=not self.verbose)
        for epoch in range(self.num_steps):
            loss_total = 0.0
            n = 0.0
            for x, y in loader:

                self.optimizer.zero_grad()
                val, _ = self.model(x)
                loss = F.mse_loss(val, y, reduction="mean")
                loss.backward()
                self.optimizer.step()

                n += 1
                loss_total += loss.item()

            pbar.set_description(f"Loss: {loss_total/n:.4f}")
            pbar.update(1)

        self.model.eval()
        self.save()
        self.train_t += 1

    def reset(self):
        """
        Resets the model
        """
        return None

    @property
    def metric(self):
        return make_metric(self.model)

    def embed(self, x):
        """
        Returns an embedding if model has one else
        returns input
        """
        if x.device != self.device:
            x = x.to(self.device)

        if hasattr(self.model, "embed"):
            return self.model.embed(x)
        return x
