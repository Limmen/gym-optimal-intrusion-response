import numpy as np

def running_average(x, N):
    """
    Function used to compute the running average
        of the last N elements of a vector x
    :param x: the vector
    :param N: N
    :return: running average
    """
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = [x[-1]]
    return y[-1]


def running_average_list(x, N):
    """
    Function used to compute the running average
        of the last N elements of a vector x and return the list of running averages
    :param x: the vector
    :param N: N
    :return: list of running averages
    """
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

