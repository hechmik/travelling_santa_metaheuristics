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


def total_length_loop(r, c, *args):
    """
    Just total length of a tour without penalties,
    starting and ending at the same city
    here 0 does not need to be at the start,
    since we do not need to keep track of non-prime 10th steps

    :param r: route
    :param c: cities
    :return:
    """

    # cities sorted by route
    c = c[r, :]

    # shifted cities array by -1. The first city becomes the last, the second becomes the first, etc.
    cs = np.roll(c, -1, axis=0)

    # compute distance for each step
    d = np.sqrt((c[:, 1] - cs[:, 1]) ** 2 + (c[:, 2] - cs[:, 2]) ** 2)

    return np.sum(d)


def total_length_straight(r, c, *args):
    """
    Just total length of a tour without penalization,
    starting and ending at different cities
    here 0 does not need to be at the start,
    since we do not need to keep track of non-prime 10th steps

    :param r: route
    :param c: cities
    :param args:
    :return:
    """

    # cities sorted by route
    c = c[r, :]

    # shifted cities array by -1. The first city becomes the last, the second becomes the first, etc.
    cs = np.roll(c, -1, axis=0)

    # remove last row, corresponding to return to first city
    cs = cs[:-1]
    c = c[:-1]

    # compute distance for each step
    d = np.sqrt((c[:, 1] - cs[:, 1]) ** 2 + (c[:, 2] - cs[:, 2]) ** 2)

    return np.sum(d)


def edp(r, c, black_list, *args):
    """
    Euclidean Distance with Penalties:
    every 10th city is penalized if it is inside the black list.
    The City IDS (first column of c) must be sorted!

    :param r: route
    :param c: cities (the city IDS must be sorted!)
    :param black_list: black list of cities to penalize, if they are on tenth steps
    :return: the total distance of the route, counting penalities
    """

    # check if there are 0s at start and end
    # r must be processed in order to have a 0 at the start only
    if r[0] != 0:
        r = np.concatenate(([0], r))
    if r[-1] == 0:
        r = r[:-1]

    # cities sorted by route
    c = c[r, :]

    # shifted cities array by -1. The first city becomes the last, the second becomes the first, etc.
    cs = np.roll(c, -1, axis=0)

    # compute distance for each step
    d = np.sqrt((c[:, 1] - cs[:, 1]) ** 2 + (c[:, 2] - cs[:, 2]) ** 2)

    # city IDS array
    cid = c[:, 0].astype(int)

    # tenth steps
    idx = np.arange(9, len(c), 10)

    # cities ids of tenth steps
    pc = cid[idx]

    # tenth steps in the black list, e.g. not prime tenth steps
    sel = idx[black_list[pc]]

    # penalize tenth steps in the black list
    d[sel] *= 1.1

    return np.sum(d)


def edp_unordered_loop(r, c, black_list):
    """
     Euclidean Distance with Penalties, for route starting and ending at the same city
     every 10th city is penalized if it is inside the black list
     The City IDS need not be sorted

     :param r: route
     :param c: cities (the city IDS must be sorted!)
     :param black_list: black list of cities to penalize, if they are on tenth steps
     :return: the total distance of the route, counting penalities, starting and ending at the same city
    """

    # new sorting index
    n_id = np.arange(len(c))[:, np.newaxis]

    # add column to array with the new index
    c = np.concatenate((n_id, c), 1)

    # cities sorted by route
    c = c[r, :]

    # shifted cities array by -1. The first city becomes the last, the second becomes the first, etc.
    cs = np.roll(c, -1, axis=0)

    # compute distance for each step
    d = np.sqrt((c[:, 2] - cs[:, 2]) ** 2 + (c[:, 3] - cs[:, 3]) ** 2)  # step distances array

    # city IDS array
    cid = c[:, 1].astype(int)

    # tenth steps:
    # starting from 8, as this will be used for paths not starting from 0, but that will be prefixed with 0
    idx = np.arange(8, len(c), 10)

    # cities ids of tenth steps
    pc = cid[idx]

    # tenth steps in the black list, e.g. not prime tenth steps
    sel = idx[black_list[pc]]

    # penalize tenth steps in the black list
    d[sel] *= 1.1

    return np.sum(d)


def edp_unordered_straight(r, c, black_list):
    """
    Euclidean Distance with Penalties, for route starting and ending at different cities
    every 10th city is penalized if it is inside the black list
    The City IDS need not be sorted

    :param r: route
    :param c: cities (the city IDS must be sorted!)
    :param black_list: black list of cities to penalize, if they are on tenth steps
    :return: the total distance of the route, counting penalities, starting and ending at the same city
    """

    # new sorting index
    n_id = np.arange(len(c))[:, np.newaxis]

    # add column to array with the new index
    c = np.concatenate((n_id, c), 1)

    # cities sorted by route
    c = c[r, :]

    # shifted cities array by -1. The first city becomes the last, the second becomes the first, etc.
    cs = np.roll(c, -1, axis=0)

    # delete last rows that close the loop
    cs = cs[:-1]
    c = c[:-1]

    # compute distance for each step
    d = np.sqrt((c[:, 2] - cs[:, 2]) ** 2 + (c[:, 3] - cs[:, 3]) ** 2)  # step distances array

    # city IDS array
    cid = c[:, 1].astype(int)

    # tenth steps:
    # starting from 8, as this will be used for paths not starting from 0, but that will be prefixed with 0
    idx = np.arange(8, len(c), 10)

    # cities ids of tenth steps
    pc = cid[idx]

    # tenth steps in the black list, e.g. not prime tenth steps
    sel = idx[black_list[pc]]

    # penalize tenth steps in the black list
    d[sel] *= 1.1

    return np.sum(d)
