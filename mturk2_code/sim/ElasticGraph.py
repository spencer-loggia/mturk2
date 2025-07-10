import numpy as np
import torch
from matplotlib import pyplot as plt
import math
import copy
from neurotools import util


"""
The purpose of this function is to run one iteration of relaxation, given a parameterized potential function.
"""


def deltaD(vals, dists, theta_1, theta_2, theta_3, initial_distance, dev="cpu", plot_track=True):
    vals = vals.to("cuda")
    dists = dists.to("cuda")
    theta_1 = theta_1.to("cuda")
    theta_2 = theta_2.to("cuda")
    theta_3 = theta_3.to("cuda")
    initial_distance = initial_distance.to("cuda")
    step_size = .25
    v = vals.flatten()
    inds = torch.triu_indices(len(v), len(v), offset=1)
    v = v.unsqueeze(0)
    v_dist = (v.T @ v)[inds[0], inds[1]]
    d = initial_distance
    track = []
    for i in range(10):
        step = - theta_1 * (dists - d) - theta_2 * dists * v_dist * torch.pow(torch.e, -1 * theta_3 * dists)
        dists = dists + step_size * step
        # track.append(torch.max(torch.abs(step)).item())
    #plt.plot(track)
    #plt.show()
    return dists.to("cpu")


def _optimize_dists(dists, vals, params):
    dists = deltaD(vals, dists, *params)
    return dists


def elastic_step(dists, vals, params):
    """
    param: coords: Tensor <examples * (example - 1) / 2>
    param: vals: Tensor <examples,>
    param: potential: function(distances, values)
    """
    dists = _optimize_dists(dists, vals, params)
    return dists.detach()


def value_update(target_coord, dist, vals, instR, lr, kernel_sigma=1.):
    """
    target_coord: Tensor <dims,>
    coord: Tensor <examples, dims>
    instR: Tensor <1,>
    lr: Tensor <1,>
    """
    with torch.no_grad():
        normal = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([kernel_sigma]))
        # square dists
        sdist = util.triu_to_square(dist, n=math.ceil(math.sqrt(len(dist) * 2)))[0]
        i_dist = sdist[target_coord]
        try:
            mod = 1 - normal.log_prob(torch.tensor([0.]))  # log(1/prob(center))
            update_coef = torch.log(lr) + mod + normal.log_prob(i_dist)
        except Exception as e:
            print(e)
            return [-1]
        delta_v = torch.exp(update_coef) * instR
        vals = vals + delta_v
        return vals.detach().cpu()


class ElasticEmbed:

    def __init__(self, ambient_dims, init_coords, theta_1=1., theta_2=1., theta_3=.1, lr=.1,
                 track=False, compile_opt=True, dev="cpu"):
        self.dims = ambient_dims
        self.vals = torch.zeros(len(init_coords), device=dev)
        self.theta_1 = theta_1.to(dev)
        self.theta_2 = theta_2.to(dev)
        self.theta_3 = theta_3.to(dev)
        self.initial_distances = torch.pdist(init_coords).flatten().to(dev)
        self.dists = self.initial_distances.clone().to(dev)
        self.lr = lr.to(dev)
        self.track = track
        self.val_hist = []
        self.dist_hist = []

        self.fail = False

        if track:
            self.val_hist.append(self.vals.detach().cpu().clone())
            self.dist_hist.append(self.dists.detach().cpu().clone())

    def get_val(self, loc):
        return self.vals[loc]

    def relax(self):
        if self.fail:
            return
        params = (self.theta_1.detach().clone(),
                  self.theta_2.detach().clone(),
                  self.theta_3.detach().clone(),
                  self.initial_distances.detach().clone())
        self.dists = elastic_step(self.dists, self.vals, params)
        if self.track and (len(self.val_hist) + 1) % 50 == 0:
            self.dist_hist.append(self.dists.detach())

    def update(self, reward, loc):
        if self.fail:
            return
        with torch.no_grad():
            v = value_update(loc, self.dists, self.vals, reward, self.lr)
            if v[0] == -1:
                print("t1:", self.theta_1, "t2", self.theta_2, "t2", "lr", self.lr)
                self.vals *= 0. # set values to 0.
                self.fail = True
            else:
                self.vals = v
            if self.track:
                self.val_hist.append(self.vals.detach().clone())
