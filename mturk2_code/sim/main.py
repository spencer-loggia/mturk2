import math

import numpy as np
import kernel_rl
from matplotlib import pyplot as plt, animation
import pandas as pd
import pickle
import sys


if __name__=="__main__":
    EPOCHS = int(sys.argv[3])
    BATCH = int(sys.argv[2])
    EXT = "TEST_2D_dynamic_kernel"
    SUBJECT = sys.argv[1]
    DEV = sys.argv[4]

    title = "_".join([SUBJECT, EXT, str(EPOCHS), str(BATCH)])
    print("starting", title)

    plt.switch_backend('Qt5Agg')
    subject_trials = "../saved_data/all_" + SUBJECT + ".csv"
    data = pd.read_csv(subject_trials, sep="\t")
    model = kernel_rl.KernelRL(resolution=36, device=DEV, expected_plateu=100000, kernel_mode="logistic")
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
    param_im = [[ax1.imshow(states[si], vmin=vmin, vmax=vmax)] for si in range(0, len(states), math.ceil(np.log2(max(2, BATCH // 100))))]
    ani = animation.ArtistAnimation(fig1, param_im, interval=10, blit=True, repeat_delay=100)

    model.to("cpu")
    with open("../saved_data/out_models/" + title + ".pkl", "wb") as f:
        pickle.dump(model, f)

    plt.show()

