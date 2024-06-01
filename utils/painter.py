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


def draw(ys, x=None, offset=0, xlabel='epoch', ylabel='y', ytags=None, title="default"):
    if x is None:
        x = torch.arange(start=1, end=len(ys[0]) + 1, dtype=int)
    for i in range(len(ys)):
        plt.plot(x[offset:], ys[i][offset:])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ytags is not None:
        plt.legend(ytags)
    plt.title(title)
    plt.show()


def draw_sub(ys, x=None):
    n = len(ys)
    if x is None:
        x = torch.arange(start=1, end=len(ys[0])+1, dtype=int)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)
    ax[0][0].set_title('loss of g&d')
    ax[0][1].set_title('w_distance')
    ax[1][0].set_title('test mse')
    ax[1][1].set_title('test acc')

    ax[0][0].plot(x, ys[0])
    ax[0][0].plot(x, ys[1])
    ax[0][1].plot(x, ys[2])
    ax[1][0].plot(x, ys[3])
    ax[1][1].plot(x, ys[4])

    ax[0][0].legend(['loss_g', 'loss_d'])
    plt.show()
