import numpy as np
import santas_path


def create_particle(particle_size, x_min=0, x_max=4):
    particle = x_min + (x_max - x_min) * np.random.uniform(low=0, high=1, size=particle_size)
    return particle


def create_particle_velocities(particle_size, v_min=-4, v_max=4):
    velocities = v_min + (v_max - v_min) * np.random.uniform(low=0, high=1, size=particle_size)
    return velocities


def create_pop_particles(pop_size, particle_length):
    pop = [create_particle(particle_length) for i in range(0, pop_size)]
    return np.array(pop)


def create_pop_velocities(popSize, particle_length):
    pop_vel = [create_particle_velocities(particle_length) for i in range(0, popSize)]
    return np.array(pop_vel)


def generate_santas_path_from_particles_pop(population):
    path_populations = np.argsort(population) + 1
    return path_populations


def evaluate_paths(paths, cities, not_primes_bool):
    distances = []
    for path in paths:
        distances.append(santas_path.compute_route_length(path, cities, not_primes_bool))
    return np.array(distances)


def swap_mutation(pop):
    pos_to_swap = np.random.choice(range(0, len(pop)), size=2, replace=False)
    pop[pos_to_swap[0]], pop[pos_to_swap[1]] = pop[pos_to_swap[1]], pop[pos_to_swap[0]]
    return pop


def mutate_pop_elements(pop, ro):
    elements_to_mutate = np.random.choice(range(0, ro), size=round(ro / 10), replace=False)
    # print(elements_to_mutate)
    for el in elements_to_mutate:
        pop[el] = swap_mutation(pop[el])


def update_inertia_pop_vel(best_particle, best_pop, c1, c2, decrement_factor, inertia_weight, pop, r1, r2, vel):
    inertia_weight = inertia_weight * decrement_factor
    if inertia_weight < 0.4:
        inertia_weight = 0.4
    vel = inertia_weight * vel
    vel = vel + c1 * r1 * (best_pop - pop)
    vel = vel + c2 * r2 * (best_particle - pop)
    pop = pop + vel
    return inertia_weight, pop, vel


def shuffle_worst_particle(distances, pop):
    # Shuffle worst element
    current_iteration_worst_index = np.argsort(distances)[-1]
    # print(current_iteration_worst_index)
    np.random.shuffle(pop[current_iteration_worst_index])


def initialize_parameters(cities, path_length, ro, not_primes_bool):
    # iteration counter
    t = 0
    # Initialization
    pop = create_pop_particles(pop_size=ro, particle_length=path_length)
    best_pop = pop
    vel = create_pop_velocities(popSize=ro, particle_length=path_length)
    paths = generate_santas_path_from_particles_pop(pop)
    distances = evaluate_paths(paths, cities, not_primes_bool)
    best_distances = distances
    # Initialize support variables where best results will be stored
    best_iteration = t
    best_particle_index = np.argsort(distances)[0]
    best_particle = pop[best_particle_index]
    global_min_distance = distances[best_particle_index]
    r1 = np.random.uniform(low=0, high=1)
    r2 = np.random.uniform(low=0, high=1)
    return best_distances, best_iteration, best_particle, best_pop, distances, global_min_distance, pop, r1, r2, t, vel


def particle_swarm_optimization(path_length,
                                cities,
                                ro=30,
                                number_of_iterations=1000,
                                decrement_factor=0.975,
                                c1=2,  # cognitive param
                                c2=2,  # social param
                                inertia_weight=0.99,
                                wait_interval=40):
    np_not_prime = np.vectorize(santas_path.not_prime)
    np_prime = np.vectorize(santas_path.is_prime)
    nums = np.arange(0, len(cities))
    not_primes_bool = np_not_prime(nums)

    best_distances, best_iteration, best_particle, best_pop, distances, global_min_distance, pop, r1, r2, t, vel = \
        initialize_parameters(cities, path_length, ro, not_primes_bool)

    result = [[t, global_min_distance]]
    for t in range(1, number_of_iterations + 1):
        if t % 10 == 0:
            print("Iteration: {}, minimum distance so far: {}".format(t, global_min_distance))
        # shuffle_worst_particle(distances, pop)
        inertia_weight, pop, vel = update_inertia_pop_vel(best_particle, best_pop, c1, c2, decrement_factor,
                                                          inertia_weight, pop, r1, r2, vel)
        mutate_pop_elements(pop, ro)
        # Compute new path and the related distances
        paths = generate_santas_path_from_particles_pop(pop)
        distances = evaluate_paths(paths, cities, not_primes_bool)
        # Find which paths lead to an improved distance and update the corresponding values
        distance_comparison = distances < best_distances
        best_distances[distance_comparison] = distances[distance_comparison]
        best_pop[distance_comparison] = pop[distance_comparison]
        # Update global best results if necessary
        current_iteration_best_index = np.argsort(distances)[0]
        current_min_distance = distances[current_iteration_best_index]
        if current_min_distance < global_min_distance:
            best_iteration = t
            best_particle = pop[current_iteration_best_index]
            global_min_distance = current_min_distance
        elif t > best_iteration + wait_interval:
            break
        result.append([t, global_min_distance])
    return result


def hybrid_particle_swarm_optimization(path_length,
                                       cities,
                                       ro=30,
                                       number_of_iterations=1000,
                                       decrement_factor=0.975,
                                       c1=2,  # cognitive param
                                       c2=2,  # social param
                                       inertia_weight=0.99,
                                       wait_interval=40):
    np_not_prime = np.vectorize(santas_path.not_prime)
    np_prime = np.vectorize(santas_path.is_prime)
    nums = np.arange(0, len(cities))
    not_primes_bool = np_not_prime(nums)

    best_distances, best_iteration, best_particle, best_pop, distances, global_min_distance, pop, r1, r2, t, vel = \
        initialize_parameters(cities, path_length, ro, not_primes_bool)

    result = [[t, global_min_distance]]
    for t in range(1, number_of_iterations + 1):
        if t % 50 == 0:
            print("Iteration: {}, minimum distance so far: {}".format(t, global_min_distance))
        # shuffle_worst_particle(distances, pop)
        inertia_weight, pop, vel = update_inertia_pop_vel(best_particle, best_pop, c1, c2, decrement_factor,
                                                          inertia_weight, pop, r1, r2, vel)
        mutate_pop_elements(pop, ro)
        # Compute new path and the related distances
        paths = generate_santas_path_from_particles_pop(pop)
        distances = evaluate_paths(paths, cities, not_primes_bool)
        # Find which paths lead to an improved distance and update the corresponding values
        distance_comparison = distances < best_distances
        best_distances[distance_comparison] = distances[distance_comparison]
        best_pop[distance_comparison] = pop[distance_comparison]
        # Update global best results if necessary
        current_iteration_best_index = np.argsort(distances)[0]
        current_min_distance = distances[current_iteration_best_index]
        if current_min_distance < global_min_distance:
            best_iteration = t
            best_particle = pop[current_iteration_best_index]
            global_min_distance = current_min_distance
        elif t > best_iteration + wait_interval:
            break
        # Hybrid approach: update velocities and pop params of the worst elements
        pop[int(ro / 2):] = pop[0:int(ro / 2)]
        vel[int(ro / 2):] = vel[0:int(ro / 2)]
        result.append([t, global_min_distance])
    return result
