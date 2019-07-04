import matplotlib.pyplot as plt
import numpy as np


def edp(r, c, black_list, *args):
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
    sel = idx[black_list[pc]]
    d[sel] *= 1.1

    return np.sum(d)


def edp_unordered_loop(r, c, black_list):
    n_id = np.arange(len(c))[:, np.newaxis]  # new_index
    c = np.concatenate((n_id, c), 1)
    c = c[r, :]
    cs = np.roll(c, -1, axis=0)[:-1]
    c = c[:-1]
    cid = c[:, 1].astype(int)
    d = np.sqrt((c[:, 2] - cs[:, 2]) ** 2 + (c[:, 3] - cs[:, 3]) ** 2)
    idx = np.arange(8, len(c), 10)
    pc = cid[idx]
    sel = idx[black_list[pc]]
    d[sel] *= 1.1

    return np.sum(d)


def edp_unordered_straight(r, c, black_list):
    n_id = np.arange(len(c))[:, np.newaxis]  # new_index
    c = np.concatenate((n_id, c), 1)
    c = c[r, :]
    cs = np.roll(c, -1, axis=0)[:-1]
    c = c[:-1]
    cid = c[:, 1].astype(int)
    d = np.sqrt((c[:, 2] - cs[:, 2]) ** 2 + (c[:, 3] - cs[:, 3]) ** 2)
    idx = np.arange(8, len(c), 10)
    pc = cid[idx]
    sel = idx[black_list[pc]]
    d[sel] *= 1.1

    return np.sum(d)


def shift_mutation(perm):
    n = len(perm)
    i = np.random.choice(n, 2, replace = False)
    i = np.sort(i)
    i0 = i[0]
    i1 = i[1]
    perm = np.concatenate((perm[i1:], perm[i0:i1], perm[:i0][::-1]))
    return perm


def swap_mutation(perm, *args):
    n = len(perm)
    i = np.random.choice(n, 2, replace = False)
    perm[i] = perm[i[::-1]]
    return perm


def reverse_mutation(perm, *args):
    n = len(perm) - 1
    i = np.random.choice(n, 1)[0]
    perm[i:i+2] = perm[i:i+2][::-1]
    return(perm)


def swap_primes_mutation(sub, perm, black_list, scale, n):
    l = len(perm)
    cids = sub[:, 0].astype(int)
    bool_mask = black_list[cids[perm]]
    tenths = np.arange(8, l, 10)
    p1 = bool_mask.astype(int)
    p1[tenths] *= scale
    p1[p1 == 0] = 1
    p1 = (p1 / sum(p1))
    idx1 = np.random.choice(np.arange(l), n, replace=False, p=p1)

    p2 = (~bool_mask).astype(int)

    p2[np.delete(np.arange(l), tenths)] *= scale
    p2[p2 == 0] = 1

    p2[idx1] = 0

    p2 = (p2 / sum(p2))
    idx2 = np.random.choice(np.arange(l), n, replace=False, p=p2)

    newperm = perm.copy()
    newperm[idx1] = perm[idx2]
    newperm[idx2] = perm[idx1]

    return newperm


def reverse_primes_mutation(sub, perm, black_list, scale, n):
    l = len(perm)
    cids = sub[:, 0].astype(int)
    bool_mask1 = black_list[cids[perm]]
    bool_mask2 = ~np.roll(bool_mask1, -1)
    bool_mask = bool_mask1 & bool_mask2

    tenths = np.arange(8, l, 10)

    p1 = bool_mask.astype(int)
    p1[tenths] *= scale
    p1[p1 == 0] = 1
    p1[np.arange(1, l, 2)] = 0

    p1 = p1[:-1]
    p1 = (p1 / sum(p1))
    i = np.random.choice(np.arange(l - 1), n, replace=False, p=p1)

    newperm = perm.copy()
    newperm[i] = perm[i + 1]
    newperm[i + 1] = perm[i]

    return newperm


def SA(array, fit_fun,  mut_fun, black_list, scale, n_to_mute,
       maxIter = np.inf, maxIterNoChange=200, tmin=0.01, alpha=0.999,
       perm_init=None, t_init=1000, verbose=False):

    # initialize solution
    if perm_init is None:
        perm = np.random.permutation(range(1, len(array)))
    else:
        perm = perm_init.copy()

    # init temperature
    tem = t_init

    dist = fit_fun(perm, array, black_list)  # objective function
    best_dist = dist
    best_trace = np.array([dist])
    current_trace = np.array([dist])
    best_perm = perm.copy()

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

        best_trace = np.concatenate((best_trace, [best_dist]))
        current_trace = np.concatenate((current_trace, [dist]))
        tem *= alpha

        if (citer % 100 == 0) and verbose:
            print('Iter: {}, IterNoChange: {}, Current: {}, Best: {}'.format(citer, iterNoChange, dist, best_dist))

        citer += 1
        iterNoChange += 1
        if (iterNoChange >= maxIterNoChange) or (citer >= maxIter):
            break

        # res = list(route=path, traceBest = traceBest, trace = traceCurrentLength)
    return best_perm, best_trace, current_trace


def plot_SA(b, c, n):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(0, len(c), n)
    l2 = ax.plot(x, b[::n])[0]
    l3 = ax.plot(x, c[::n])[0]

    line_labels = ["Best", "Current"]

    fig.legend([l2, l3], line_labels, bbox_to_anchor=(0.85, 0.25))

    plt.show()