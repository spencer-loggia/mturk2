from mturk2_code.sim.dataset import ColorShapeData
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.rc('font', family='monospace')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import os


def plot_reward_space(data: ColorShapeData):
    reward_vals = data.rewards.reshape((36, 36))
    plt.imshow(reward_vals, cmap='hot')
    plt.show()


def heatmap_4D_surf(fig, axs, reward_array: np.ndarray, choice_val_array: np.ndarray, ax_size=36):
    """
    :param reward_array: np.ndarray (num_agents, space_h, space_w)
    :param choice_val_array: np.ndarray (num_agents, space_h, space_w)
    """
    # as plot_surface needs 2D arrays as input
    x = np.arange(ax_size)
    y = np.arange(ax_size)
    # we make a meshgrid from the x,y data
    X, Y = np.meshgrid(x, y)

    for i in range(len(axs)):
        axs[i] = fig.gca(projection='3d')
        Z = choice_val_array[i]

        # map the data to rgba values from a colormap
        colors = cm.ScalarMappable(cmap="hot").to_rgba(reward_array[i])

        # plot_surface with points X,Y,Z and data_value as colors
        surf = axs[i].plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
                               linewidth=0, antialiased=True)

    fig.show()


def heatmap_scatterplot(ax, reward_array: np.ndarray, choice_idx, choice_count, is_best_choice: np.ndarray, ax_size=36):
    """
    visualize 3 continous and one catagorical dimmnsions on x, y, dot_size, dot_color
    """
    img = zoom(reward_array, zoom=1, order=1)
    ax.imshow(img, cmap='hot')
    choice_idx = np.array(choice_idx, dtype=int).T
    color = ['green' if best else 'cyan' for best in is_best_choice]
    label_desc = ['Best Choice' if best else 'Not Best Choice' for best in is_best_choice]
    ax.scatter(choice_idx[0], choice_idx[1], s=np.array(choice_count) * 3 + 1, c=color, label=label_desc)
    # ax.legend()

