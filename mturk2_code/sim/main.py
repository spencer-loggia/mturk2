import math
import argparse
import os.path


import numpy as np
import torch

import kernel_rl
from matplotlib import pyplot as plt, animation
import pandas as pd
import pickle
import sys
from neurotools import util, support


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="KernelRL")
    parser.add_argument("name", type=str, choices=["Sally", "Tina", "Yuri", "Buzz"])
    parser.add_argument("-l", "--load", nargs=1, type=str, action="store")
    parser.add_argument("-t", "--train", nargs=2, type=int, action="append")
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    if args.train:
        EPOCHS = args.train[0][1]
        BATCH = args.train[0][0]
    else:
        EPOCHS = None
        BATCH = None
    EXT = "2D_dynamic_kernel"
    RES = 36

    SUBJECT = args.name
    if args.cuda:
        DEV = "cuda"
    else:
        DEV = "cpu"

    title = "_".join([SUBJECT, EXT, str(EPOCHS), str(BATCH)])
    print("starting", title)
    figure_out_dir = "../saved_data/out_rl_figs/"

    plt.switch_backend('Qt5Agg')
    subject_trials = "../saved_data/all_" + SUBJECT + ".csv"
    data = pd.read_csv(subject_trials, sep="\t")

    if args.load:
        out_path = args.load[0]
        with open(args.load[0], "rb") as f:
            model = pickle.load(f)
    else:
        out_path = "../saved_data/out_models/" + title + ".pkl"
        model = kernel_rl.KernelRL(resolution=RES, device=DEV, expected_plateu=100000, kernel_mode="logistic")

    if args.train:
        model.fit(data.head(BATCH), epochs=EPOCHS, lr=.02)
    res = model.get_results()

    print("Final Params:")
    for k in res.keys():
        try:
            if len(res[k].flatten()) < 10:
                print(k, ":", res[k])
        except Exception as e:
            print(e)

    fig, ax = plt.subplots(1)
    ax.plot(res["epoch_loss_history"])

    fig0, ax0 = plt.subplots(1)
    ax0.imshow(res["initial_value_space"])

    fig1, ax1 = plt.subplots()
    states = np.array(res["state_spaces"])
    vmin = np.min(states)
    vmax = np.max(states)
    # ax.imshow(param_states[0], vmin=vmin, vmax=vmax)
    print("state min max", vmin, vmax)
    step_size = math.ceil(np.log2(max(2, len(states) // 100)))
    state_im = [states[si] for si in range(0, len(states), step_size)]
    param_im = [[ax1.imshow(s, vmin=vmin, vmax=vmax)] for s in state_im]
    ani = animation.ArtistAnimation(fig1, param_im, interval=10, blit=True, repeat_delay=100)
    state_out = os.path.join(figure_out_dir, "state_anim_" + title + ".avi")
    support.save_video_from_arrays(state_im, vmin, vmax, state_out)

    fig2, ax2 = plt.subplots()
    in_cov = torch.from_numpy(res["init_kernel_cov"])
    f_cov = torch.from_numpy(res["final_kernel_cov"])
    tau = torch.from_numpy(res["kernel_log_tau"])
    cov_t = util.exponential_func(torch.arange(0, len(states), step_size), in_cov.flatten(), f_cov.flatten(), tau)
    kernels = [util.gaussian_kernel((RES, RES), cov=c.reshape((2, 2)) * RES, renormalize=True).detach().numpy() for c in cov_t]
    vmin = np.min(np.stack(kernels))
    vmax = np.max(np.stack(kernels))
    k_im = [[ax2.imshow(k, vmin=vmin, vmax=vmax)] for k in kernels]
    ani = animation.ArtistAnimation(fig2, k_im, interval=10, blit=True, repeat_delay=100)
    state_out = os.path.join(figure_out_dir, "kernel_anim_" + title + ".avi")
    support.save_video_from_arrays(kernels, vmin, vmax, state_out)

    model.to("cpu")
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    plt.show()

