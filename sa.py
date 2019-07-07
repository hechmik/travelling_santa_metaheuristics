import matplotlib.pyplot as plt
import numpy as np


def shift_mutation(perm):
    """
    Performs a shift mutation on a permutation
    """
    n = len(perm)
    i = np.random.choice(n, 2, replace = False)
    i = np.sort(i)
    i0 = i[0]
    i1 = i[1]
    perm = np.concatenate((perm[i1:], perm[i0:i1], perm[:i0][::-1]))
    return perm


def swap_mutation(perm, *args):
    """
    Performs a swap mutation on a permutation
    """
    n = len(perm)
    i = np.random.choice(n, 2, replace = False)
    perm[i] = perm[i[::-1]]
    return perm


def reverse_mutation(perm, *args):
    """
    Performs a reverse mutation on a permutation
    """
    n = len(perm) - 1
    i = np.random.choice(n, 1)[0]
    perm[i:i+2] = perm[i:i+2][::-1]
    return perm


def swap_primes_mutation(sub, perm, black_list, scale, n):
    """
    Performs a swap mutation on a permutation.
    Cities on the black list have a higher chance of being swapped with cities not on the blacklist,
    if the former are on tenth steps.

    Not used in the final project, it led to no improvement. Maybe a bug?
    """
    l = len(perm)
    cids = sub[:, 0].astype(int)

    # boolean mask of city ids, ordered by the permutation, on the black list
    bool_mask = black_list[cids[perm]]

    # index of tenth steps:
    # starting from 8, as this will be used for paths not starting from 0, but that will be prefixed with 0
    tenths = np.arange(8, l, 10)

    # -- computation of probability of selecting the first batch of cities
    # the boolean mask becomes a binary string (the 1s are cities on the black list)
    p1 = bool_mask.astype(int)
    # The 1s corresponding to cities on the black list are assigned a higher weight
    p1[tenths] *= scale
    # every other city is assigned a small weight
    p1[p1 == 0] = 1
    # the weights are normalized to become probabilities
    p1 = (p1 / sum(p1))

    # n cities are selected, according to their weight
    idx1 = np.random.choice(np.arange(l), n, replace=False, p=p1)

    # -- computation of probabilities of selecting the second batch of cities
    # negation of the first boolean mask, i.e., cities not on the black list, typed as a binary array
    p2 = (~bool_mask).astype(int)
    # The 1s corresponding to cities not on the black list and that are not tenth steps are assigned a higher weight
    p2[np.delete(np.arange(l), tenths)] *= scale
    # every other city is assigned a small weight
    p2[p2 == 0] = 1
    # cities selected in the first batch are assigned a 0 probability, so that there will not be doubles
    p2[idx1] = 0
    # the weights are normalized to become probabilities
    p2 = (p2 / sum(p2))
    # n cities are selected according to this new probabilies
    idx2 = np.random.choice(np.arange(l), n, replace=False, p=p2)

    # the cities are swapped
    newperm = perm.copy()
    newperm[idx1] = perm[idx2]
    newperm[idx2] = perm[idx1]

    return newperm


def reverse_primes_mutation(sub, perm, black_list, scale, n):
    """
    Performs a swap mutation on a permutation.
    Cities on the black list have a higher chance of being  with cities not on the blacklist,
    if the former are on tenth steps
    """
    l = len(perm)

    cids = sub[:, 0].astype(int)

    # boolean mask of cities on the black list
    bool_mask1 = black_list[cids[perm]]
    # boolean mask of cities whose successor in the route is not on the black list
    bool_mask2 = ~np.roll(bool_mask1, -1)
    # conjunction of the two boolean masks:
    # cities on the black list whose successor is not on the black list
    bool_mask = bool_mask1 & bool_mask2

    # index of tenth steps:
    # starting from 8, as this will be used for paths not starting from 0, but that will be prefixed with 0
    tenths = np.arange(8, l, 10)

    # the boolean mask becomes a binary array
    p1 = bool_mask.astype(int)
    # 1s, corresponding to cities on the black list whose successor is not on the black list, are given a higher weight
    p1[tenths] *= scale
    # every other city is given a smaller weight
    p1[p1 == 0] = 1
    # if n > 1, some indices are set as 0 in order to not make overlapping reverses
    if n > 1:
        p1[np.arange(1, l, 2)] = 0

    # the last element of the weight list is deleted,
    #  as the corresponding city does not have a successor to be swapped with
    p1 = p1[:-1]
    # the weights are normalized to become probabilities
    p1 = (p1 / sum(p1))
    # n cities are selected according to this probability
    i = np.random.choice(np.arange(l - 1), n, replace=False, p=p1)

    # the n cities selected are swapped with their successor
    newperm = perm.copy()
    newperm[i] = perm[i + 1]
    newperm[i + 1] = perm[i]

    return newperm


def SA(array, fit_fun,  mut_fun, black_list, scale, n_to_mute,
       maxIter = np.inf, maxIterNoChange=200, tmin=0.01, alpha=0.999,
       perm_init=None, t_init=1000, verbose=False):
    """
    Simple implementation of the SA algorithm

    :param array: array, in our case the cities array
    :param fit_fun: fitness function
    :param mut_fun: mutation function
    :param black_list: black list
    :param scale: scale to assign a higher weight to cities in the black list
    :param n_to_mute: how many elements should be mutated
    :param maxIter: max number of iteration, np.inf by default to run until convergence
    :param maxIterNoChange: max number of iterations for convergence
    :param tmin: min temperature
    :param alpha: constant to update temperature
    :param perm_init: initial permutation
    :param t_init: initial temperature
    :param verbose: if True, the progress of the algorithm is printed
    :return: the best permutation found, and the traces of the best permutations and the current permutations
    """

    # initialize solution
    if perm_init is None:
        perm = np.random.permutation(range(1, len(array)))
    else:
        perm = perm_init.copy()

    # init temperature
    tem = t_init

    dist = fit_fun(perm, array, black_list)  # objective function
    best_dist = dist  # init best dist
    best_perm = perm.copy()   # init best permutation

    citer = 0
    iterNoChange = 0

    while tem >= tmin:

        # mutate
        newperm = mut_fun(array, perm, black_list, scale, n_to_mute)
        dist_new = fit_fun(newperm, array, black_list)

        if dist_new <= best_dist:
            perm = newperm
            dist = dist_new
            best_dist = dist
            best_perm = perm.copy()
            iterNoChange = 0
        elif np.exp((dist - dist_new) / tem) > np.random.uniform():
            dist = dist_new
            perm = newperm
            iterNoChange = 0

        # update traces
        tem *= alpha

        if (citer % 100 == 0) and verbose:
            print('Iter: {}, IterNoChange: {}, Current: {}, Best: {}'.format(citer, iterNoChange, dist, best_dist))

        citer += 1
        iterNoChange += 1
        if (iterNoChange >= maxIterNoChange) or (citer >= maxIter):
            break

    return best_perm


