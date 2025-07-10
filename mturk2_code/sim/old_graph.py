import numpy as np
import torch
from matplotlib import pyplot as plt
import math
import copy


"""
The purpose of this function is to run one iteration of relaxation, given a parameterized potential function.
"""


def neighbor_distances(coords):
    device = coords.device

    # figure out n, m, and a flattened coords array
    if coords.ndim == 3:
        n, _, m = coords.shape
        coords_flat = coords.view(n * n, m)
    elif coords.ndim == 2:
        n2, m = coords.shape
        n = int(math.isqrt(n2))
        assert n * n == n2, f"cannot infer square grid from length={n2}"
        coords_flat = coords
    else:
        raise ValueError(f"coords must be 2- or 3-D, got shape {coords.shape}")

    # 8 neighbor offsets in grid space
    offsets = torch.tensor([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1],
    ], device=device)

    # build (n2,2) array of grid coordinates
    y, x = torch.meshgrid(
        torch.arange(n, device=device),
        torch.arange(n, device=device),
        indexing='ij'
    )
    grid = torch.stack([y, x], dim=-1).view(n * n, 2)  # [n2,2]

    # add each offset to get neighbor positions, clamp to [0,n-1]
    nbr_xy = (grid.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, n - 1)  # [n2,8,2]

    # flatten to indices into coords_flat
    nbr_idx = nbr_xy[:, :, 0] * n + nbr_xy[:, :, 1]  # [n2,8]

    # lookup neighbor coords and compute distances
    nbr_coords = coords_flat[nbr_idx]  # [n2,8,m]
    base_coords = coords_flat.unsqueeze(1).expand(-1, 8, -1)  # [n2,8,m]

    # L2 distances
    dists = torch.norm(nbr_coords - base_coords, dim=-1)  # [n2,8]
    return dists


def potential(dists, val_dists, theta_1, theta_2, theta_3, initial_distance, dev="cpu"):
    theta_1 = theta_1.to(dev)
    theta_2 = theta_2.to(dev)
    theta_3 = theta_3.to(dev)
    initial_distance = initial_distance.detach().clone().to(dev)
    return (theta_1 * torch.square(dists - initial_distance) +
            (theta_2 * (val_dists - theta_3) * torch.square(dists)))


def _optimize_coords(coords, vals, params):
    optim = torch.optim.Adam([coords], lr=0.2, eps=1e-7)

    for _ in range(50):
        optim.zero_grad()
        dists = neighbor_distances(coords).flatten()
        val_dists = neighbor_distances(vals[:, None]).flatten().detach()
        potentials = potential(dists, val_dists, *params, dev=dists.device)
        energy = torch.sum(potentials)
        energy.backward()
        optim.step()

    return coords, energy


def elastic_step(vals, coords, params, eval=False):
    """
    param: coords: Tensor <examples, dims>
    param: vals: Tensor <examples,>
    param: potential: function(distances, values)
    """
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    coords = coords.to(dev).detach().clone().requires_grad_(True).to(dev)
    vals = vals.to(dev)
    coords, energy = _optimize_coords(coords, vals, params)
    return coords.detach().cpu()


def value_update(target_coord, coords, vals, instR, lr, kernel_sigma=1):
    """
    target_coord: Tensor <dims,>
    coord: Tensor <examples, dims>
    instR: Tensor <1,>
    lr: Tensor <1,>
    """
    with torch.no_grad():
        dist = torch.distributions.MultivariateNormal(loc=target_coord,
                                                      covariance_matrix=torch.eye(len(target_coord)) * kernel_sigma)
        mod = 1 - dist.log_prob(target_coord)  # log(1/prob(center))
        update_coef = torch.log(lr) + mod + dist.log_prob(coords)
        delta_v = torch.exp(update_coef) * instR
        vals = vals + delta_v
        return vals.detach().cpu()



class ElasticEmbed:

    def __init__(self, ambient_dims, init_coords, theta_1=1., theta_2=1., theta_3=0., val_init=.5, lr=.1, track=False, compile_opt=True, dev="cpu"):
        self.dims = ambient_dims
        self.vals = torch.ones(len(init_coords)) * val_init
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.coords = init_coords
        self.initial_distances = neighbor_distances(init_coords).flatten()
        self.lr = lr
        self.track = track
        self.val_hist = []
        self.coord_hist = []
        if compile_opt:
            torch.compile(_optimize_coords, fullgraph=True) # compile function
        if track:
            self.val_hist.append(self.vals.detach().clone())
            self.coord_hist.append(self.coords.detach().clone())

    def get_val(self, loc):
        return self.vals[loc]

    def relax(self):
        self.coords = self.coords.detach()
        self.vals = self.vals.detach()
        params = (self.theta_1.detach().clone(),
                  self.theta_2.detach().clone(),
                  self.theta_3.detach().clone(),
                  self.initial_distances.detach().clone())
        self.coords = elastic_step(self.vals, self.coords, params)
        if self.track:
            self.coord_hist.append(self.coords.detach())

    def update(self, reward, loc):
        with torch.no_grad():
            self.coords = self.coords.detach()
            self.vals = self.vals.detach()
            coordinate = self.coords[loc].squeeze()
            self.vals = value_update(coordinate, self.coords, self.vals, reward, self.lr)
            if self.track:
                self.val_hist.append(self.vals.detach().clone())
