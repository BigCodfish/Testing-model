import torch
from matplotlib import pyplot as plt


def draw(y, x=None, xlabel='x', ylabel='y'):
    if x is None:
        x = torch.arange(start=1, end=len(y) + 1, dtype=int)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def draw(ys, x=None, offset=0, xlabel='x', ylabel='y', ytags=None, title="default", sub=None):
    if x is None:
        x = torch.arange(start=1, end=len(ys[0]) + 1, dtype=int)
    for i in range(len(ys)):
        if sub is None:
            plt.plot(x[offset:], ys[i][offset:])
        else:
            sub.plot(x[offset:], ys[i][offset:])
    if sub is None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ytags is not None:
            plt.legend(ytags)
        plt.title(title)
        plt.show()
    else:
        sub.legend(ytags)

def draw_sub(ys, x=None):
    n = len(ys)
    if x is None:
        x = torch.arange(start=1, end=len(ys[0])+1, dtype=int)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)
    ax[0][0].set_title('loss of g&d')
    ax[1][0].set_title('w_distance')
    ax[1][1].set_title('loss test')

    ax[0][0].plot(x, ys[0])
    ax[0][0].plot(x, ys[1])
    ax[1][0].plot(x, ys[2])
    ax[1][1].plot(x, ys[3])

    ax[0][0].legend(['loss_g', 'loss_d'])

    plt.show()