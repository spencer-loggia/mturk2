import numpy as np
import torch
from matplotlib import pyplot as plt
import math
import copy
from neurotools import util
import igraph


"""
The purpose of this function is to run one iteration of relaxation, given a parameterized potential function.
"""


def log_beta_function(alpha):
    """
    Compute log of the multivariate beta function B(alpha)
    for batch input: shape [..., K]
    """
    return torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha.sum(dim=-1))


def log_dirichlet_multinomial(x, alpha):
    """
    Compute log P(x | alpha) for batched x, alpha of shape [..., K]
    """
    x = x.to(dtype=torch.float)
    alpha = alpha.to(dtype=torch.float)

    n = x.sum(dim=-1)
    log_coeff = torch.lgamma(n + 1) - torch.lgamma(x + 1).sum(dim=-1)
    log_beta = log_beta_function(x + alpha) - log_beta_function(alpha)

    return log_coeff + log_beta


def multinomial_samesource(a, b, alpha, prior_H0):
    """
    compute P(same dist | a, b) for multiple pairs.

    Args:
        a (Tensor): <batch, K> counts
        b (Tensor): <Batch, K> counts
        alpha (Tensor): [K] or [B, K] dirichlet prior
        prior_H0 (Tensor): <B,> prior probabilities that H0 is true.

    Returns:
        Tensor: [B] posterior probabilities that H1 is true (same source)
    """
    a = a.to(dtype=torch.float)
    b = b.to(dtype=torch.float)
    prior_H0 = prior_H0.to(dtype=torch.float)
    prior_H1 = 1.0 - prior_H0

    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0).expand_as(a)

    log_p_H0 = log_dirichlet_multinomial(a, alpha) + log_dirichlet_multinomial(b, alpha)

    ab = a + b
    log_coeff_ab = (
        torch.lgamma(a.sum(dim=-1) + 1) - torch.lgamma(a + 1).sum(dim=-1) +
        torch.lgamma(b.sum(dim=-1) + 1) - torch.lgamma(b + 1).sum(dim=-1)
    )
    log_p_H1 = log_coeff_ab + log_beta_function(ab + alpha) - log_beta_function(alpha)

    log_prior_H1 = torch.log(prior_H1)
    log_prior_H0 = torch.log(prior_H0)

    log_num = log_p_H1 + log_prior_H1
    log_denom = torch.logaddexp(log_num, log_p_H0 + log_prior_H0)

    return torch.exp(log_num - log_denom)


class ElasticEmbed:

    def __init__(self, n_reward_levels, init_coords, theta_1=1., theta_2=1., lr=.1,
                 track=False, reward_override=None, compile_opt=True, dev="cpu"):
        self.theta_1 = theta_1.to(dev)
        self.theta_2 = theta_2.to(dev)
        self.n_reward_levels = n_reward_levels
        self.dev = dev

        # updated by update
        self.val_counts_n = torch.ones((len(init_coords), n_reward_levels), device=dev)  # frequency of getting each value level at each node (weighted by assoc probability)
        self.val_counts = torch.ones(n_reward_levels, device=dev)
        self.vals = torch.zeros((len(init_coords), n_reward_levels), device=dev) + 1 / n_reward_levels

        if reward_override is None:
            self.reward = torch.arange(n_reward_levels, device=dev)
        else:
            self.reward = reward_override.to(dev)

        # probability that two nodes are associated
        self.prior_assoc = util.triu_to_square(1 - torch.sigmoid(self.theta_1 * torch.pdist(init_coords).flatten()),
                                               n=len(init_coords)).squeeze()
        self.assoc = self.prior_assoc.clone()

        self.lr = lr.to(dev)
        self.track = track
        self.val_hist = []
        self.dist_hist = []

        self.fail = False

        if track:
            self.val_hist.append(self.vals.detach().clone())
            self.dist_hist.append(self.assoc.detach().clone())

    def get_val(self, loc):
        # return the weighted probability of each value
        # sum of relevant edges
        v_w = self.assoc[loc]
        uv = self.vals.T @ v_w.T # <>
        v = uv.squeeze() / torch.sum(v_w)
        v = torch.sum(v * self.reward.unsqueeze(1), dim=0)  # weight by subjective value perception
        return v  # <1>

    def relax(self):
        if self.fail:
            return
        # each value should be compared with each other
        v = torch.tile(self.vals.unsqueeze(1), (1, len(self.vals), 1))
        inds = torch.triu_indices(len(self.vals), len(self.vals), offset=1)
        a_counts = v[inds[0], inds[1], :]
        b_counts = v.transpose(0, 1)[inds[0], inds[1], :]
        prior = 1 - self.prior_assoc[inds[0], inds[1]] # prior belief on non-existance of assoc.
        alpha = self.theta_2 * torch.ones((self.n_reward_levels), device=self.dev)  # controls resistance to change.
        bayes_assoc = multinomial_samesource(a=a_counts, b=b_counts, alpha=alpha, prior_H0=prior)
        self.assoc = util.triu_to_square(bayes_assoc, n=len(self.vals)).squeeze()

    def update(self, obs_r: int, loc):
        # simple value frequency distribution update.
        # TODO: Still using a RL-ey value update here that should be replaced post VSS
        if self.fail:
            return
        self.val_counts[obs_r] += 1
        v_w = self.assoc[loc]
        self.val_counts_n[:, obs_r] += v_w * self.lr
        self.vals = self.val_counts_n / (torch.sum(self.val_counts_n, dim=1, keepdim=True) + 1e-10)
