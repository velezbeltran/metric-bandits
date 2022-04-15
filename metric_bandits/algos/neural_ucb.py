"""
The implementation of neural UCB follows very closely what is described in
the paper. Name of variables are chosen so as to agree with the paper.
`https://arxiv.org/pdf/1911.04462.pdf`
s T, regularization parameter λ, exploration parameter ν, confidence parameter δ, norm
parameter S, step size η, number of gradient descent steps J, network width m, network depth L.
"""
import random
from collections import defaultdict
from math import sqrt

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metric_bandits.algos.base import BaseAlgo
from metric_bandits.utils.math import sherman_morrison
from metric_bandits.utils.nn import make_metric


class NeuralUCB(BaseAlgo):
    def __init__(
        self,
        model,
        step_size,
        num_steps,
        train_freq,
        explore_param,
        active=False,
        verbose=True,
    ):
        """
        If active is true, the model forgets completely about regret and just takes actions
        with the aim of maximizing information gain.
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
        self.Z_inv = None
        self.t = 0

        # parameters to keep track of
        self.active = active
        self.last_action = None
        self.rewards = []
        self.contexts_played = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        self.model.eval()
        if self.active:
            return self.choose_action_active(actions)
        else:
            return self.choose_action_default(actions)

    def choose_action_default(self, actions):
        self.ucb_val_grads, self.ucb_estimate = {}, {}
        for action in actions:
            val, grad = self.get_val_grad(actions[action])
            self.ucb_val_grads[action] = (val, grad)
            opt = self.optimist_reward(grad)
            self.ucb_estimate[action] = val + opt

        # return the key with the highest value
        self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        self.last_grad = self.ucb_val_grads[self.last_action][1]
        self.contexts_played.append(actions[self.last_action])
        return self.last_action

    def choose_action_active(self, actions):
        """
        Chooses an action based on the current state of the model. The pair that is chossen
        is the one that has the highest gradient. Currently only works in the implementation
        where a similarity is provided at the end and there are two pairs.

        This is not the cleanest way of doing this but it probably is the easiest one at the moment.
        """
        self.ucb_val_grads = defaultdict(list)
        self.ucb_estimate = defaultdict(int)
        self.unique_contexts = defaultdict(list)

        # Keep track of relevant values per unique context

        # We shuffle in case we have an optimist reward of zero
        items = list(actions.items())
        random.shuffle(items)
        for action, ctxt in items:
            ctxt = str(ctxt[:-1])
            val, grad = self.get_val_grad(actions[action])
            self.ucb_val_grads[ctxt].append((val, grad))
            self.ucb_estimate[ctxt] += self.optimist_reward(grad)
            self.unique_contexts[ctxt].append(action)

        # Choose to make a desicion on the pair with the highes opt value
        self.last_context = max(self.ucb_estimate, key=self.ucb_estimate.get)
        # choose the action with the highest value
        ctxt_val = self.ucb_val_grads[self.last_context]
        argmax = 0 if ctxt_val[0] > ctxt_val[1] else 1
        self.last_action = self.unique_contexts[self.last_context][argmax]
        self.last_grad = ctxt_val[argmax][1]
        self.contexts_played.append(actions[self.last_action])
        return self.last_action

    def update(self, reward):
        """
        Updates the model
        """
        self.rewards.append(reward)

        # update our confidence matrix
        prev_grad = self.last_grad / sqrt(self.model.num_params)
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

    def reset(self):
        """
        Resets the model
        """
        self.Z_inv = torch.eye(
            self.model.num_params, requires_grad=False, device=self.device
        )
        print("Reset model")

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
