import numpy as np
import math


def not_prime(n):
    if n == 2:
        return False
    if n % 2 == 0 or n <= 1:
        return True

    sqr = int(math.sqrt(n)) + 1

    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return True
    return False


def is_prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False

    sqr = int(math.sqrt(n)) + 1

    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return False
    return True


def total_length(r, c):
    c = c[r, :]
    cs = np.roll(c, -1, axis=0)
    d = np.sqrt((c[:, 1] - cs[:, 1]) ** 2 + (c[:, 2] - cs[:, 2]) ** 2)

    return np.sum(d)


def total_length_w_penalties(r, c, black_list):
    # v8
    # every 10th city is penalized if it is inside the black list
    # check if there are 0s at start and end
    # r must be processed in order to have a 0 at the start
    if r[0] != 0:
        r = np.concatenate(([0], r))
    if r[-1] == 0:
        r = r[:-1]
    c = c[r, :]
    cs = np.roll(c, -1, axis=0)
    cid = c[:, 0].astype(int)
    d = np.sqrt((c[:, 1] - cs[:, 1]) ** 2 + (c[:, 2] - cs[:, 2]) ** 2)
    idx = np.arange(9, len(c), 10)
    pc = cid[idx]

    sel = (idx)[black_list[pc]]
    d[sel] *= 1.1

    return np.sum(d)
