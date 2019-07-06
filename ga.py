import numpy as np


def route_fitness(r, c, length_fun, **kwargs):
    """
    fitness of a route
    """
    black_list = kwargs.get("black_list")
    return 1 / length_fun(r, c, black_list)


def subset_fitness(perm, c, subs, length_fun, **kwargs):
    """
    fitness of a route, when dealing with ordering subsets/clusters
    """
    route = np.concatenate(subs[perm])
    black_list = kwargs.get("black_list")
    return 1 / length_fun(route, c)


def pop_gen(n, pop_size, include_zero = True):
    """
    Random population generation
    """
    if include_zero:
        s = 0
    else:
        s = 1
    arr = np.repeat([np.arange(s, n)], pop_size, 0)
    return np.apply_along_axis(np.random.permutation, 1, arr)


def pop_eval(array, pop, fit_fun, **kwargs):
    """
    Evaluation of population by the fitness function
    """
    length_fun = kwargs.get("length_fun")
    black_list = kwargs.get("black_list")
    subs = kwargs.get("subs")
    return np.apply_along_axis(fit_fun, 1, pop, c = array, length_fun = length_fun, black_list = black_list, subs = subs)


def pop_stats(scores):
    """
    Population stats
    """
    mean = np.mean(scores)
    med = np.median(scores)
    best = np.max(scores)
    return med, mean, best


def roulette_selection(pop, scores, size):
    """
    Roulette Selection
    """
    if size % 2 != 0:
        size += 1
    p = scores/np.sum(scores)
    sel = np.random.choice(len(pop), size, replace = True, p = p).reshape((size//2, 2))
    return pop[sel]


def two_point_crossover(p):
    """
    Two Point Crossover, to be applied directly to a parents array returned by the roulette selection phase
    """
    n = p.shape[2]
    swath = np.random.randint(1, n)

    cut1 = np.random.randint(0, n - swath)
    cut2 = cut1 + swath

    ps0 = p.shape[0]
    off = np.zeros((ps0, n))

    sel = p[:, 0, cut1:cut2]
    off[:, cut1:cut2] = sel

    p1 = p[:, 1]
    sel = np.array(np.split(p1[~np.array([(np.isin(p1[i], sel[i])) for i in range(len(sel))])], ps0))

    off[:, 0:cut1] = sel[:, 0:cut1]
    off[:, cut2:] = sel[:, cut1:]
    return off


def mod_two_point_crossover(p):
    """
    Modified Two Point Crossover:
    http://www.rroij.com/open-access/enhanced-order-crossover-for-permutationproblems.php?aid=50178
    """
    n = p.shape[2]
    swath = np.random.randint(1, n // 7)

    cut1 = np.random.randint(0, n - swath)
    cut2 = cut1 + swath

    # print(cut1, cut2)

    ps0 = p.shape[0]
    off = np.zeros((ps0, n))

    sel = p[:, 0, cut1:cut2]
    off[:, cut1:cut2] = sel

    p1 = p[:, 1]
    sel = np.array(np.split(p1[~np.array([(np.isin(p1[i], sel[i])) for i in range(len(sel))])], ps0))

    off[:, 0:cut1] = sel[:, 0:cut1]
    off[:, cut2:] = sel[:, cut1:]
    return off


def two_point_crossover3(p):
    """
    Alternative Version of Two-Point crossover
    """
    n = p.shape[2]
    swath = np.random.randint(1, n)
    cut1 = np.random.randint(0, n - swath)
    cut2 = cut1 + swath
    ps0 = p.shape[0]

    off = np.zeros((ps0, n))
    sel = p[:, 0, cut1:cut2]
    off[:, cut1:cut2] = sel
    p1 = p[:, 1]

    result = np.empty_like(p1)
    for i in range(ps0):
        result[i] = ~np.isin(p1[i], sel[i])
    result = result.astype(bool)

    sel = np.array(np.split(p1[result], ps0))

    off[:, 0:cut1] = sel[:, 0:cut1]
    off[:, cut2:] = sel[:, cut1:]
    return off


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


def pop_mutation(pop, mut_fun, mut_perc):
    """
    Apply mutation function to percentage of the population
    """
    sel = np.random.choice(len(pop), int(len(pop) * mut_perc), replace = False)
    pop[sel] = np.apply_along_axis(mut_fun, 1, pop[sel])
    return pop


def GA(array, n_gen, pop_size, parent_size, fit_fun, mut_funs, mut_perc=0.1, sel_fun=roulette_selection,
       cross_fun=two_point_crossover, pop_include_zero=True, max_no_change=100, on_subsets=False, verbose=False, **kwargs):
    """
    Simple implementation of the GA algorithm

    :param array: e.g. cities array
    :param n_gen: total number of generations. np.inf by default
    :param pop_size: size of population
    :param parent_size: how many parents to select. Offsprings will be half of parent_size
    :param fit_fun: fitness function
    :param mut_funs: list of mutation functions to use
    :param mut_perc: percentage of population to mutate
    :param sel_fun: selection function
    :param cross_fun: crossover function
    :param pop_include_zero: whether the permutations in the population should include the point 0
    :param max_no_change: maximum number of iterations with no change of best memeber before convergence
    :param on_subsets: if True, GA is applied to find order of clusters/subsets
    :param verbose:  if True, prints the progress of the algorithm
    :param kwargs: Trace of median, mean and best fitness values, overall best value, and best overall permutation
    :return:
    """
    subs = kwargs.get("subs")
    if on_subsets:
        n = len(subs)
    else:
        n = len(array)

    # init pop
    pop = pop_gen(n, pop_size, pop_include_zero)

    length_fun = kwargs.get("length_fun")
    black_list = kwargs.get("black_list")

    scores = pop_eval(array, pop, fit_fun, length_fun=length_fun, black_list=black_list, subs=subs)
    med, mean, best = pop_stats(scores)

    ov_best = best  # overall best score init
    best_pop = pop  # overall best pop init
    med_trace = np.array([med])  # trace of median scores init
    mean_trace = np.array([mean])  # trace of mean scores init
    best_trace = np.array([best])  # trace of best scores init

    iter_no_change = 0
    i = 0

    while i < n_gen:
        # pass n_gen = np.inf to iterate until iter_no_change >= max_no_change

        # select best parents
        parents = sel_fun(pop, scores, parent_size)

        # generate offsprings
        offs = cross_fun(parents)

        # replace worst in pop with offs
        sort = np.argsort(scores)
        pop[sort[:len(offs)]] = offs

        # mutate pop
        for fun in mut_funs:
            pop = pop_mutation(pop, fun, mut_perc)

        # evaluate
        # scores = pop_eval(array, pop, fit_fun)
        scores = pop_eval(array, pop, fit_fun, length_fun=length_fun, black_list=black_list, subs=subs)

        # update traces
        med, mean, best = pop_stats(scores)  # new med, mean best scores

        if best > ov_best:
            ov_best = best
            best_pop = pop.copy()
            iter_no_change = 0

        med_trace = np.concatenate((med_trace, [med]))
        mean_trace = np.concatenate((mean_trace, [mean]))
        best_trace = np.concatenate((best_trace, [best]))

        if verbose:
            if i % 1000 == 0:
                print('Iter {}, ItNoChange {}, Best {}'.format(i, iter_no_change, 1 / ov_best))

        i += 1
        iter_no_change += 1
        if iter_no_change >= max_no_change:
            break

    # best_scores = pop_eval(array, best_pop, fit_fun)
    best_scores = pop_eval(array, best_pop, fit_fun, length_fun=length_fun, black_list=black_list, subs=subs)
    bpe = best_pop[np.argmax(best_scores)]

    return med_trace, mean_trace, best_trace, ov_best, bpe