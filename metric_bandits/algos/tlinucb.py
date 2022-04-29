import random
from collections import defaultdict

import torch

from metric_bandits.algos.base import BaseAlgo
from metric_bandits.utils.math import get_argmax, sherman_morrison


class TLinUCB(BaseAlgo):
    def __init__(
        self,
        dim_input,
        explore_param,
        reg=1.0,
        active=False,
        verbose=True,
    ):
        """
        If active is true, the model forgets completely about regret and just takes actions
        with the aim of maximizing information gain.
        """
        super().__init__()
        self.explore_param = explore_param
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.dim_input = dim_input
        self.reg = reg

        # state of the algorithm
        self.Z_inv = None
        self.t = 0
        self.train_t = 0

        # parameters to keep track of
        self.active = active
        self.last_action = None
        self.rewards = []

    def choose_action(self, actions):
        """
        actions is a list-like object dictionary and contains the available actions
        """
        if self.active:
            return self.choose_action_active(actions)
        else:
            return self.choose_action_default(actions)

    def choose_action_default(self, actions):
        self.ucb_val_opts, self.ucb_estimate = {}, {}
        for action in actions:
            ctx = actions[action][:, :-1]  # last element is the action
            val = self.theta.T @ ctx
            opt = self.optimist_reward(ctx)
            self.ucb_val_opts[action] = (val, opt)
            self.ucb_estimate[action] = val + opt

        # return the key with the highest value
        self.last_action = max(self.ucb_estimate, key=self.ucb_estimate.get)
        self.last_context = actions[self.last_action][:, :-1]
        return self.last_action

    def choose_action_active(self, actions):
        """
        Chooses an action based on the current state of the model. The pair that is chossen
        is the one that has the highest gradient. Currently only works in the implementation
        where a similarity is provided at the end and there are two pairs.

        This is not the cleanest way of doing this but it probably is the easiest one at the moment.
        """
        self.ucb_estimate = defaultdict(int)
        self.unique_contexts = defaultdict(list)
        self.ucb_val_opts = defaultdict(list)

        # Keep track of relevant values per unique context

        # We shuffle in case we have an optimist reward of zero
        items = list(actions.items())
        random.shuffle(items)
        for action, ctxt in items:
            ctxt_str = str(ctxt[:, :-1])
            ctxt = ctxt[:, :-1]
            val = self.theta.T @ ctxt
            opt = self.optimist_reward(ctxt)
            self.ucb_val_opts[ctxt_str].append((val, opt))
            self.ucb_estimate[ctxt_str] += opt
            self.unique_contexts[ctxt_str].append(action)

        # Choose to make a desicion on the pair with the highes opt value
        self.last_context_str = max(
            self.ucb_estimate, key=self.ucb_estimate.get
        ).detach()

        # choose the action with the highest value
        ctxt_val_opt = self.ucb_val_opts[self.last_context_str]
        argmax = get_argmax(ctxt_val_opt, lambda x: x[0])
        self.last_action = self.unique_contexts[self.last_context_str][argmax]
        self.last_context = actions[self.last_action][:, :-1]
        return self.last_action

    def optimist_reward(self, context):
        """
        Returns the optimist reward for a given context.
        """
        return self.explore_param * torch.sqrt(context.T @ self.Z_inv @ context)

    def update(self, reward):
        """
        Updates the model
        """
        print("in update context shape:", self.last_contex.shape)
        self.rewards.append(reward)

        # update our confidence matrix
        self.Z_inv = sherman_morrison(self.Z_inv, self.last_context)
        self.b = self.b + self.last_context * reward

        # decide whether to train the model
        self.t += 1

    @property
    def theta(self):
        return self.Z_inv @ self.b

    def reset(self):
        """
        Resets the model
        """
        self.Z_inv = self.reg * torch.eye(
            self.dim_input**2, requires_grad=False, device=self.device
        )
        self.b = torch.zeros(
            (1, self.dim_input**2), requires_grad=False, device=self.device
        )
