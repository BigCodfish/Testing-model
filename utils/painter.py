import os.path

import pandas as pd
import torch
from matplotlib import pyplot as plt


# def draw(y, x=None, xlabel='x', ylabel='y'):
#     if x is None:
#         x = torch.arange(start=1, end=len(y) + 1, dtype=int)
#     plt.plot(x, y)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.show()

fig_save_path = 'C:\\Users\\39228\Desktop\\figsave'

def draw(ys, x=None, offset=0, xlabel='epoch', ylabel='y', ytags=None, title="default", save_name=None):
    if x is None:
        x = torch.arange(start=1, end=len(ys[0]) + 1, dtype=int)
    for i in range(len(ys)):
        plt.plot(x[offset:], ys[i][offset:])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ytags is not None:
        plt.legend(ytags)
    plt.title(title)
    if save_name is not None:
        save_path = os.path.join(fig_save_path, save_name)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def draw_sub(ys, titles, x=None):
    n = len(ys) // 2
    if x is None:
        x = torch.arange(start=1, end=len(ys[0])+1, dtype=int)
    fig, ax = plt.subplots(2, n)
    fig.set_size_inches(6*n, 10)
    ax[0][0].set_title('loss of g&d')
    ax[0][0].plot(x, ys[0])
    ax[0][0].plot(x, ys[1])
    ax[0][0].legend(['loss_g', 'loss_d'])
    i = 0
    for j in range(2, len(ys)):
        if j > n:
            i = 1
            j -= n
        ax[i][j-1].plot(x, ys[i * n + j])
        ax[i][j-1].set_title(titles[i * n + j])
    plt.show()
