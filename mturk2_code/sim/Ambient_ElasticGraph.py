import numpy as np
import torch
from matplotlib import pyplot as plt
import math
import copy


"""
This one assumes stable perceptual distances, with value as an additional dimension. Value can be mixed in distance 
computations according to a parameter. Bayes parameters include:
- Rad C
- Rad S
- Value Mix Coef.
- LR
- Temperature
- Min Reward 
- Max Reward 
"""

def deltaD(vals, dists, theta_1, theta_2, initial_distance, dev="cpu", plot_track=True):
    step_size = .7
    v = vals.flatten()
    inds = torch.triu_indices(len(v), len(v), offset=1)
    v = v.unsqueeze(0)
    v_dist = (v.T @ v)[inds[0], inds[1]]
    d = initial_distance
    track = []
    for i in range(5):
        step = -(dists - d) - theta_1 * dists * v_dist * torch.pow(torch.e, -1 * theta_2 * dists)
        dists = dists + step_size * step
        track.append(torch.max(torch.abs(step)).item())
    plt.plot(track)
    plt.show()
    return dists


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

    def __init__(self, ambient_dims, init_coords, theta_1=1., theta_2=1., theta_3=0., theta_4=2., theta_5=0., val_init=.5, lr=.1,
                 track=False, compile_opt=True, dev="cpu"):
        self.dims = ambient_dims
        self.vals = torch.ones(len(init_coords)) * theta_3
        self.percept_coords = init_coords
        self.values = torch.ones(len(self.percept_coords)) * val_init
        self.val_mix = theta_1
        self.lr = lr
        self.track = track
        self.val_hist = []
        self.dist_hist = []

        if track:
            self.val_hist.append(self.vals.detach().clone())
            self.dist_hist.append(self.dists.detach().clone())

    def get_val(self, loc):
        return self.vals[loc]

    def relax(self):
        params = (self.theta_1.detach().clone(),
                  self.theta_2.detach().clone(),
                  self.initial_distances.detach().clone())
        self.dists = elastic_step(self.dists, self.vals, params)
        if self.track and (len(self.val_hist) + 1) % 50 == 0:
            self.dist_hist.append(self.dists.detach())

    def update(self, reward, loc):
        with torch.no_grad():
            v = value_update(loc, self.dists, self.vals, reward, self.lr)
            if v[0] == -1:
                print("t1:", self.theta_1, "t2", self.theta_2, "t2", "lr", self.lr)
            else:
                self.vals = v
            if self.track:
                self.val_hist.append(self.vals.detach().clone())