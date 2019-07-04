import numpy as np
import santas_path
import random


def create_particle(particle_size, x_min=0, x_max=4):
    particle = x_min + (x_max - x_min) * np.random.uniform(low=0, high=1, size=particle_size)
    return particle


def create_particle_velocities(particle_size, v_min=-4, v_max=4):
    velocities = v_min + (v_max - v_min) * np.random.uniform(low=0, high=1, size=particle_size)
    return velocities


def create_pop_particles(pop_size, particle_length):
    pop = [create_particle(particle_length) for i in range(0, pop_size)]
    return np.array(pop)


def create_pop_velocities(pop_size, particle_length):
    pop_vel = [create_particle_velocities(particle_length) for i in range(0, pop_size)]
    return np.array(pop_vel)


def generate_santas_path_from_particles_pop(population):
    path_populations = np.argsort(population)
    return path_populations


def generate_santas_path_from_particles_pop_complete_dataset(population):
    return generate_santas_path_from_particles_pop(population) + 1


def evaluate_paths(paths, cities, not_primes_bool):
    distances = []
    for path in paths:
        distances.append(santas_path.total_length_w_penalties(path, cities, not_primes_bool))
    return np.array(distances)


def swap_mutation(pop):
    pos_to_swap = np.random.choice(range(0, len(pop)), size=2, replace=False)
    pop[pos_to_swap[0]], pop[pos_to_swap[1]] = pop[pos_to_swap[1]], pop[pos_to_swap[0]]
    return pop


def mutate_pop_elements(pop, ro):
    elements_to_mutate = np.random.choice(range(0, ro), size=round(ro / 10), replace=False)
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
    np.random.shuffle(pop[current_iteration_worst_index])


def initialize_parameters(cities, path_length, ro, not_primes_bool):
    # iteration counter
    t = 0
    # Initialization
    pop = create_pop_particles(pop_size=ro, particle_length=path_length)
    best_pop = pop
    vel = create_pop_velocities(pop_size=ro, particle_length=path_length)
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
                                wait_interval=200,
                                hybrid_evolutionary_approach=False):
    np_not_prime = np.vectorize(santas_path.not_prime)
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
        paths = generate_santas_path_from_particles_pop_complete_dataset(pop)
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
        if hybrid_evolutionary_approach:
            pop[int(ro / 2):] = pop[0:int(ro / 2)]
            vel[int(ro / 2):] = vel[0:int(ro / 2)]
        result.append([t, global_min_distance])
    return result


def compute_final_paths(paths, clusters_els):
    end_paths = []
    clusters_els = np.array(clusters_els)
    for path in paths:
        current_path = []
        path = np.array(path)
        for cluster_path in clusters_els[path]:
            current_path = np.concatenate((current_path, cluster_path))
        current_path = current_path.astype(int)
        end_paths.append(current_path)
    return end_paths


def evaluate_paths_cluster(paths, cities, not_primes_bool):
    distances = []
    for path in paths:
        distances.append(santas_path.total_length(path, cities))
    return np.array(distances)


def generate_santas_path_from_particles_cluster(population, cluster_elements):
    path_populations = []
    for particle in population:
        path = np.sort(cluster_elements)[np.argsort(particle)]
        path = np.delete(path, np.where(path == 0))
        path_populations.append(path)
    return path_populations


def initialize_clusters(cluster_dict, ro, cluster_indexes, cities, not_primes_bool):
    clusters_pop = []
    clusters_vel = []
    clusters_best_path = []
    clusters_best_particle = []
    clusters_distances = []
    for c in cluster_dict.items():
        pop = create_pop_particles(pop_size=ro, particle_length=c[1])
        clusters_pop.append(pop)
        vel = create_pop_velocities(pop_size=ro, particle_length=c[1])
        clusters_vel.append(vel)
        paths = generate_santas_path_from_particles_cluster(pop, np.where(cluster_indexes == c[0])[0])
        distances = evaluate_paths_cluster(paths, cities, not_primes_bool)
        clusters_distances.append(distances)
        best_path_index = np.argsort(distances)[0]
        best_path = paths[best_path_index]
        clusters_best_path.append(best_path)

        clusters_best_particle.append(pop[best_path_index])
    return clusters_pop, clusters_vel, clusters_best_path, clusters_best_particle, clusters_distances


def update_cluster_parameters(c1, c2, cities, clusterized_cities, clusters_best_particle, clusters_best_path,
                              clusters_best_pop, clusters_distances, clusters_pop, clusters_vel, inertia_weight,
                              not_primes_bool, number_of_clusters, r1, r2, ro, hybrid_evolutionary_approach):
    for cluster in range(number_of_clusters):
        current_pop = clusters_pop[cluster]
        mutate_pop_elements(current_pop, ro)
        current_best_pop = clusters_best_pop[cluster]
        current_vel = clusters_vel[cluster]
        current_best_particle = clusters_best_particle[cluster]

        current_vel = current_vel * inertia_weight
        current_vel = current_vel + c1 * r1 * (current_best_pop - current_pop)
        current_vel = current_vel + c2 * r2 * (current_best_particle - current_pop)
        current_pop = current_pop + current_vel

        current_paths = generate_santas_path_from_particles_cluster(current_pop,
                                                                    np.where(clusterized_cities == cluster)[0])
        distances = evaluate_paths_cluster(current_paths, cities, not_primes_bool)
        current_dist = clusters_distances[cluster]
        current_best_dist = np.sort(current_dist)[0]
        distance_comparison = distances < current_dist
        current_dist[distance_comparison] = distances[distance_comparison]
        current_best_pop[distance_comparison] = current_pop[distance_comparison]

        if hybrid_evolutionary_approach:
            current_pop[int(ro / 2):] = current_pop[0:int(ro / 2)]
            current_vel[int(ro / 2):] = current_vel[0:int(ro / 2)]

        # Update storing structures
        clusters_best_pop[cluster] = current_best_pop
        clusters_distances[cluster] = current_dist
        clusters_pop[cluster] = current_pop
        clusters_vel[cluster] = current_vel

        # Update best particle if needed
        local_best_particle_distance = np.sort(distances)[0]
        if local_best_particle_distance < current_best_dist:
            best_particle_index = np.argsort(current_dist)[0]
            clusters_best_particle[cluster] = current_pop[best_particle_index]
            clusters_best_path[cluster] = current_paths[best_particle_index]



def generate_overall_params(cities, clusters_best_path, clusters_pop, not_primes_bool, ro):
    number_of_clusters = len(clusters_pop)
    overall_pop = create_pop_particles(pop_size=ro, particle_length=number_of_clusters)
    overall_vel = create_pop_velocities(pop_size=ro, particle_length=number_of_clusters)
    overall_paths = generate_santas_path_from_particles_pop(overall_pop)
    final_complete_paths = compute_final_paths(overall_paths, clusters_best_path)
    final_complete_distances = evaluate_paths(paths=final_complete_paths, cities=cities,
                                              not_primes_bool=not_primes_bool)
    return final_complete_distances, final_complete_paths, number_of_clusters, overall_pop, overall_vel


def generate_cluster_dict(clusterized_cities):
    cluster_dict = {}
    for cluster_index in set(clusterized_cities):
        cluster_dict[cluster_index] = np.where(clusterized_cities == cluster_index)[0].shape[0]
    return cluster_dict


def generate_not_primes_numbers_array(cities):
    np_not_prime = np.vectorize(santas_path.not_prime)
    nums = np.arange(0, len(cities))
    not_primes_bool = np_not_prime(nums)
    return not_primes_bool


def update_overall_parameters(best_final_complete_distances, best_overall_distance, best_overall_iteration,
                              best_overall_particle, best_overall_pop, c1, c2, cities, clusters_best_path, final_path,
                              inertia_weight, not_primes_bool, overall_pop, overall_vel, r1, r2, ro, t):
    mutate_pop_elements(overall_pop, ro)
    overall_vel = overall_vel * inertia_weight
    overall_vel = overall_vel + c1 * r1 * (best_overall_pop - overall_pop)
    overall_vel = overall_vel + c2 * r2 * (best_overall_particle - overall_pop)
    overall_pop = overall_pop + overall_vel
    overall_paths = generate_santas_path_from_particles_pop(overall_pop)
    final_complete_paths = compute_final_paths(overall_paths, clusters_best_path)
    final_complete_distances = evaluate_paths(paths=final_complete_paths, cities=cities,
                                              not_primes_bool=not_primes_bool)
    # Find which paths lead to an improved distance and update the corresponding values
    distance_comparison = final_complete_distances < best_final_complete_distances
    best_final_complete_distances[distance_comparison] = final_complete_distances[distance_comparison]
    best_overall_pop[distance_comparison] = overall_pop[distance_comparison]
    # Update global best results if necessary
    current_iteration_best_index = np.argsort(final_complete_distances)[0]
    current_min_distance = final_complete_distances[current_iteration_best_index]
    if current_min_distance < best_overall_distance:
        best_overall_iteration = t
        best_overall_particle = overall_pop[current_iteration_best_index]
        final_path = final_complete_paths[current_iteration_best_index]
        best_overall_distance = current_min_distance
    return best_overall_distance, best_overall_iteration, final_path, overall_vel, overall_pop, best_overall_particle, best_overall_pop


def cluster_particle_swarm_optimization(
        cities,
        clusterized_cities,
        ro=30,
        max_number_of_iterations=3000,
        decrement_factor=0.975,
        hybrid_evolutionary_approach=False,
        c1=2,  # cognitive param
        c2=2,  # social param
        inertia_weight=0.99,
        wait_interval=200):
    random.seed(10)
    np.random.seed(10)

    not_primes_bool = generate_not_primes_numbers_array(cities)

    t = 0

    cluster_dict = generate_cluster_dict(clusterized_cities)
    clusters_pop, clusters_vel, clusters_best_path, clusters_best_particle, clusters_distances = initialize_clusters(
        cluster_dict, ro, clusterized_cities, cities, not_primes_bool)
    clusters_best_pop = clusters_pop
    r1 = np.random.uniform(low=0, high=1)
    r2 = np.random.uniform(low=0, high=1)

    final_complete_distances, final_complete_paths, number_of_clusters, overall_pop, overall_vel = generate_overall_params(
        cities, clusters_best_path, clusters_pop, not_primes_bool, ro)

    best_final_complete_distances = final_complete_distances
    best_overall_pop = overall_pop
    best_overall_particle = overall_pop[np.argsort(final_complete_distances)[0]]
    final_path = final_complete_paths[np.argsort(final_complete_distances)[0]]
    best_overall_iteration = t
    best_overall_distance = np.sort(final_complete_distances)[0]

    while True:
        t = t + 1
        if t > max_number_of_iterations:
            print("Reached max number of iterations")
            break
        if t % 50 == 0:
            print("Iteration: {}, min_distance: {}".format(t, best_overall_distance))
        inertia_weight = inertia_weight * decrement_factor
        if inertia_weight < 0.4:
            inertia_weight = 0.4
        update_cluster_parameters(c1, c2, cities, clusterized_cities, clusters_best_particle, clusters_best_path,
                                  clusters_best_pop, clusters_distances, clusters_pop, clusters_vel, inertia_weight,
                                  not_primes_bool, number_of_clusters, r1, r2, ro, hybrid_evolutionary_approach)
        # Update overall particles
        best_overall_distance, best_overall_iteration, final_path, overall_vel, overall_pop, best_overall_particle, best_overall_pop = update_overall_parameters(
            best_final_complete_distances, best_overall_distance, best_overall_iteration, best_overall_particle,
            best_overall_pop, c1, c2, cities, clusters_best_path, final_path, inertia_weight, not_primes_bool,
            overall_pop, overall_vel, r1, r2, ro, t)
        # Hybrid approach: update velocities and pop params of the worst elements
        if hybrid_evolutionary_approach:
            overall_pop[int(ro / 2):] = overall_pop[0:int(ro / 2)]
            overall_vel[int(ro / 2):] = overall_vel[0:int(ro / 2)]

        if t > best_overall_iteration + wait_interval:
            print("The algorithm is no longer improving")
            break
    return best_overall_iteration, best_overall_distance, final_path


def cluster_particle_swarm_optimization_bottom_up(
        cities,
        clusterized_cities,
        ro=30,
        max_number_of_iterations=3000,
        number_of_iterations_clusters=500,
        hybrid_evolutionary_approach=False,
        decrement_factor=0.975,
        c1=2,  # cognitive param
        c2=2,  # social param
        inertia_weight=0.99,
        wait_interval=200):
    random.seed(10)
    np.random.seed(10)

    not_primes_bool = generate_not_primes_numbers_array(cities)
    t = 0

    original_inertia_weight = inertia_weight

    cluster_dict = generate_cluster_dict(clusterized_cities)
    clusters_pop, clusters_vel, clusters_best_path, clusters_best_particle, clusters_distances = initialize_clusters(
        cluster_dict, ro, clusterized_cities, cities, not_primes_bool)
    clusters_best_pop = clusters_pop
    r1 = np.random.uniform(low=0, high=1)
    r2 = np.random.uniform(low=0, high=1)
    """ To DO:
        - PSO for determining how to "combine" clusters. Easy solution: choose randomly
        - Assemble the path according to the sequence obtained in the previous step
        - Compute distance for this sequences
    """
    final_complete_distances, final_complete_paths, number_of_clusters, overall_pop, overall_vel = generate_overall_params(
        cities, clusters_best_path, clusters_pop, not_primes_bool, ro)
    best_final_complete_distances = final_complete_distances
    best_overall_pop = overall_pop
    best_overall_particle = overall_pop[np.argsort(final_complete_distances)[0]]
    final_path = final_complete_paths[np.argsort(final_complete_distances)[0]]
    best_overall_iteration = t
    best_overall_distance = np.sort(final_complete_distances)[0]

    # Apply PSO to Clusters
    for t in range(1, number_of_iterations_clusters):
        if t % 50 == 0:
            print("Iteration_Cluster: {}".format(t))
        inertia_weight = inertia_weight * decrement_factor
        if inertia_weight < 0.4:
            inertia_weight = 0.4
        update_cluster_parameters(c1, c2, cities, clusterized_cities, clusters_best_particle, clusters_best_path,
                                  clusters_best_pop, clusters_distances, clusters_pop, clusters_vel, inertia_weight,
                                  not_primes_bool, number_of_clusters, r1, r2, ro, hybrid_evolutionary_approach)

    # Now update overall params
    inertia_weight = original_inertia_weight
    t = 0
    while True:
        t = t + 1
        if t > max_number_of_iterations:
            print("Reached max number of iterations")
            break
        if t % 50 == 0:
            print("Iteration_overall: {}, min_distance: {}".format(t, best_overall_distance))
        # Update overall particles
        best_overall_distance, best_overall_iteration, final_path, overall_vel, overall_pop, best_overall_particle, best_overall_pop = update_overall_parameters(
            best_final_complete_distances, best_overall_distance, best_overall_iteration, best_overall_particle,
            best_overall_pop, c1, c2, cities, clusters_best_path, final_path, inertia_weight, not_primes_bool,
            overall_pop, overall_vel, r1, r2, ro, t)
        if t > best_overall_iteration + wait_interval:
            print("The algorithm is no longer improving")
            break

        inertia_weight = inertia_weight * decrement_factor
        if inertia_weight < 0.4:
            inertia_weight = 0.4
        overall_vel = inertia_weight * overall_vel

    return best_overall_iteration, best_overall_distance, final_path
