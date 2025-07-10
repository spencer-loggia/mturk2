import math
import argparse
import os.path


import numpy as np
import torch

#import kernel_rl
import perceptual_rl
from matplotlib import pyplot as plt, animation
import pandas as pd
import pickle
import sys
import time
#from neurotools import util, support


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="MT2RL")
    parser.add_argument("mode", type=str, choices=["Kernel", "GRU", "Elastic"])
    parser.add_argument("name", type=str, choices=["YT", "SB", "Yuri", "Tina", "Sally", "ALL"])
    parser.add_argument("-l", "--load", nargs=1, type=str, action="store")
    parser.add_argument("-t", "--train", nargs=2, type=int, action="append")
    parser.add_argument("--cuda", action="store_true")

    fit = True

    time.sleep(2.5 * 3600)

    args = parser.parse_args()

    if args.train:
        EPOCHS = args.train[0][1]
        BATCH = args.train[0][0]
    else:
        EPOCHS = None
        BATCH = None
    EXT = "bayes_dyno"
    RES = 37

    if args.name == "YT":
        SUBJECT = ["Yuri", "Tina"]
    elif args.name == "Yuri":
        SUBJECT = ["Yuri"]
    elif args.name == "Sally":
        SUBJECT = ["Sally"]
    elif args.name == "ALL":
        SUBJECT = ["Yuri", "Tina", "Buzz", "Sally"]
    if args.cuda:
        DEV = "cuda"
    else:
        DEV = "cpu"
    MODE = args.mode
    if args.mode == "kernel":
        algo = kernel_rl.KernelRL
    elif args.mode == "GRU":
        algo = perceptual_rl.PerceptRL
    elif args.mode == "nontemporal":
        algo = perceptual_rl.NTPerceptRL
    elif args.mode == "Elastic":
        algo = perceptual_rl.CSC_RL
    else:
        raise ValueError

    title = "_".join([args.name, EXT, MODE])
    print("starting", title)
    figure_out_dir = "../saved_data/out_rl_figs/"

    plt.switch_backend('Qt5Agg')
    data = []
    for s in SUBJECT:
        subject_trials = "../saved_data/all_" + s + ".csv"
        d = pd.read_csv(subject_trials, sep="\t")
        if SUBJECT == "Tina":
            d = d.iloc[8500:]
        data.append(d.head(BATCH))



    if args.load:
        out_path = args.load[0]
        with open(args.load[0], "rb") as f:
            model = pickle.load(f)
    else:
        out_dir = "../saved_data/out_models/"
        out_path = os.path.join(out_dir,  title + ".pkl")
        model = algo(resolution=RES, n_c=36, n_s=36, device=DEV, expected_plateu=110000, kernel_mode="logistic", positional_dim=4)

    fit_res = []
    if args.train and fit:
        fit_res = model.fit(data, epochs=EPOCHS, lr=.02, snap_out=out_path)
    res = model.get_results()

    print("Final Params:")
    for k in res.keys():
        try:
            print(k, ":", res[k])
        except Exception as e:
            print(e)

    # evaluate GP


    coord_hist, val_hist, prob_history = model.predict(data)
    np.save(os.path.join(out_dir, title + "_coordhist.npy"), coord_hist)
    np.save(os.path.join(out_dir, title + "_valhist.npy"), val_hist)
    np.save(os.path.join(out_dir, title + "_probhist.npy"), prob_history)
    with open(os.path.join(out_dir, title + "_params.txt"), "w") as f:
        f.write(str(res))
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


    fig, ax = plt.subplots(1)
    #ax.plot(res["epoch_loss_history"])

    if MODE == "kernel":
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

    plt.plot(fit_res)
    plt.show()

