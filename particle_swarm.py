import numpy as np
import santas_path
import random
#import queue


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
        distances.append(santas_path.edp(path, cities, not_primes_bool))
    return np.array(distances)


def swap_mutation(pop):
    pos_to_swap = np.random.choice(range(0, len(pop)), size=2, replace=False)
    pop[pos_to_swap[0]], pop[pos_to_swap[1]] = pop[pos_to_swap[1]], pop[pos_to_swap[0]]
    return pop


def mutate_pop_elements(pop, ro):
    elements_to_mutate = np.random.choice(range(0, ro), size=round(ro / 10), replace=False)
    for el in elements_to_mutate:
        pop[el] = swap_mutation(pop[el])


def particle_swarm_optimization(cities,
                                ro=30,
                                max_number_of_iterations=2000,
                                decrement_factor=0.975,
                                c1=2,  # cognitive param
                                c2=2,  # social param
                                inertia_weight=0.99,
                                wait_interval=500,
                                hybrid_evolutionary_approach=False):
    # For reproducibility
    random.seed(10)
    np.random.seed(10)

    not_primes_bool = generate_not_primes_numbers_array(cities)
    # iteration counter
    t = 0
    # Initialization
    path_length = len(cities) - 1 #Exclude 0 because it's a special node
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
    best_path = paths[best_particle_index]
    global_min_distance = distances[best_particle_index]
    r1 = np.random.uniform(low=0, high=1)
    r2 = np.random.uniform(low=0, high=1)

    results_for_each_iteration = [[t, global_min_distance]]
    for t in range(1, max_number_of_iterations + 1):
        if t % 100 == 0:
            print("Iteration: {}, minimum distance so far: {}".format(t, global_min_distance))

        vel = vel + c1 * r1 * (best_pop - pop)
        vel = vel + c2 * r2 * (best_particle - pop)
        pop = pop + vel

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
            best_path = paths[current_iteration_best_index]
        results_for_each_iteration.append([t, global_min_distance])

        if t > best_iteration + wait_interval:
            break

        # Hybrid approach: update velocities and pop params of the worst elements
        if hybrid_evolutionary_approach:
            particle_index_ordered_by_distance = np.argsort(distances)
            best_particles_index = particle_index_ordered_by_distance[0:int(ro / 2)]
            worst_particles_index = particle_index_ordered_by_distance[int(ro / 2):]
            pop[worst_particles_index] = pop[best_particles_index]
            vel[worst_particles_index] = vel[best_particles_index]

        # Update inertia and velocity
        inertia_weight = inertia_weight * decrement_factor
        if inertia_weight < 0.4:
            inertia_weight = 0.4
        vel = inertia_weight * vel

    return results_for_each_iteration, best_path


def compute_final_paths(paths, clusters_els):
    """
    Compute the complete paths taking into consideration the cluster orders and how the dataset was clusterized
    :param paths: array containing the chosen cluster combinations
    :param clusters_els: array containing the cluster related to each dataset point
    :return:
    """
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


def evaluate_paths_cluster(paths, cities):
    """
    Compute the euclidean distance for each path
    :param paths:
    :param cities:
    :return:
    """
    distances = []
    for path in paths:
        distances.append(santas_path.total_length_straight(path, cities))
    return np.array(distances)


def generate_santas_path_from_particles_cluster(population, cluster_elements):
    """
    Transform the given particles in feasible paths using the SVP rule
    :param population:
    :param cluster_elements:
    :return:
    """
    path_populations = []
    for particle in population:
        path = np.sort(cluster_elements)[np.argsort(particle)]
        # This city is the North Pole, starting and ending point of each path: it should not be considered
        path = np.delete(path, np.where(path == 0))
        path_populations.append(path)
    return path_populations


def initialize_clusters(cluster_dict, ro, cluster_indexes, cities):
    """
    Initialize the fundamental elements needed by PSO algorithm
    :param cluster_dict:
    :param ro:
    :param cluster_indexes:
    :param cities:
    :return:
    """
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
        distances = evaluate_paths_cluster(paths, cities)
        clusters_distances.append(distances)
        best_path_index = np.argsort(distances)[0]
        best_path = paths[best_path_index]
        clusters_best_path.append(best_path)

        clusters_best_particle.append(pop[best_path_index])
    return clusters_pop, clusters_vel, clusters_best_path, clusters_best_particle, clusters_distances


def initialize_overall_params(cities, clusters_best_path, clusters_pop, not_primes_bool, ro):
    """
    Initialize parameters needed by PSO for findingn the best order for assembling clusters
    :param cities:
    :param clusters_best_path:
    :param clusters_pop:
    :param not_primes_bool:
    :param ro:
    :return:
    """
    number_of_clusters = len(clusters_pop)
    overall_pop = create_pop_particles(pop_size=ro, particle_length=number_of_clusters)
    overall_vel = create_pop_velocities(pop_size=ro, particle_length=number_of_clusters)
    overall_paths = generate_santas_path_from_particles_pop(overall_pop)
    final_complete_paths = compute_final_paths(overall_paths, clusters_best_path)
    final_complete_distances = evaluate_paths(paths=final_complete_paths, cities=cities,
                                              not_primes_bool=not_primes_bool)
    return final_complete_distances, final_complete_paths, number_of_clusters, overall_pop, overall_vel


def generate_cluster_dict(clusterized_cities):
    """
    Generate a dictionary where its keys are the clusters ID and its values are the number of its elements
    :param clusterized_cities:
    :return:
    """
    cluster_dict = {}
    for cluster_index in set(clusterized_cities):
        cluster_dict[cluster_index] = np.where(clusterized_cities == cluster_index)[0].shape[0]
    return cluster_dict


def generate_not_primes_numbers_array(cities):
    """
    Find which are the cities whose ID is a prime number
    :param cities:
    :return:
    """
    np_not_prime = np.vectorize(santas_path.not_prime)
    nums = np.arange(0, len(cities))
    not_primes_bool = np_not_prime(nums)
    return not_primes_bool


def cluster_particle_swarm_optimization(
        cities,
        clusterized_cities,
        ro=30,
        max_number_of_iterations=1500,
        number_of_iterations_clusters=1500,
        hybrid_evolutionary_approach=False,
        decrement_factor=0.975,
        c1=2,  # cognitive param
        c2=2,  # social param
        inertia_weight=0.99,
        wait_interval=500):
    # For reproducibility
    random.seed(10)
    np.random.seed(10)

    not_primes_bool = generate_not_primes_numbers_array(cities)
    t = 0

    original_inertia_weight = inertia_weight

    cluster_dict = generate_cluster_dict(clusterized_cities)
    clusters_pop, clusters_vel, clusters_best_path, clusters_best_particle, clusters_distances = initialize_clusters(
        cluster_dict, ro, clusterized_cities, cities)
    clusters_best_pop = clusters_pop
    r1 = np.random.uniform(low=0, high=1)
    r2 = np.random.uniform(low=0, high=1)

    final_complete_distances, final_complete_paths, number_of_clusters, overall_pop, overall_vel = initialize_overall_params(
        cities, clusters_best_path, clusters_pop, not_primes_bool, ro)
    best_final_complete_distances = final_complete_distances
    best_overall_pop = overall_pop
    best_overall_particle = overall_pop[np.argsort(final_complete_distances)[0]]
    final_path = final_complete_paths[np.argsort(final_complete_distances)[0]]
    best_overall_iteration = t
    best_overall_distance = np.sort(final_complete_distances)[0]

    for cluster in range(number_of_clusters):
        if cluster % 100 == 0:
            print("Start working on cluster {}".format(cluster))
        best_overall_iteration = 0
        inertia_weight = original_inertia_weight

        current_pop = clusters_pop[cluster]
        current_best_pop = clusters_best_pop[cluster]
        current_vel = clusters_vel[cluster]
        current_dist = clusters_distances[cluster]
        current_best_particle = clusters_best_particle[cluster]
        current_best_path = clusters_best_path[cluster]
        for i in range(1, number_of_iterations_clusters + 1):

            mutate_pop_elements(current_pop, ro)

            current_vel = current_vel + c1 * r1 * (current_best_pop - current_pop)
            current_vel = current_vel + c2 * r2 * (current_best_particle - current_pop)
            current_pop = current_pop + current_vel

            current_paths = generate_santas_path_from_particles_cluster(current_pop,
                                                                        np.where(clusterized_cities == cluster)[0])
            distances = evaluate_paths_cluster(current_paths, cities)

            current_best_dist = np.sort(current_dist)[0]
            distance_comparison = distances < current_dist
            current_dist[distance_comparison] = distances[distance_comparison]
            current_best_pop[distance_comparison] = current_pop[distance_comparison]

            if hybrid_evolutionary_approach:
                best_elements_indexes = np.argsort(distances)[0:int(ro / 2)]
                worst_elements_indexes = np.argsort(distances)[int(ro / 2):]
                current_pop[worst_elements_indexes] = current_pop[best_elements_indexes]
                current_vel[worst_elements_indexes] = current_vel[worst_elements_indexes]

            # Update best particle if needed
            local_best_particle_distance = np.sort(distances)[0]
            if local_best_particle_distance < current_best_dist:
                best_particle_index = np.argsort(current_dist)[0]
                current_best_particle = current_pop[best_particle_index]
                #clusters_best_particle[cluster] = current_pop[best_particle_index]
                current_best_path = current_paths[best_particle_index]
                #clusters_best_path[cluster] = current_paths[best_particle_index]
                best_overall_iteration = i
            if i > best_overall_iteration + wait_interval:
                #print("The algorithm is no longer improving")
                break
            inertia_weight = inertia_weight * decrement_factor
            if inertia_weight < 0.4:
                inertia_weight = 0.4
            current_vel = current_vel * inertia_weight

        # Update storing structures
        clusters_best_pop[cluster] = current_best_pop
        clusters_distances[cluster] = current_dist
        clusters_pop[cluster] = current_pop
        clusters_vel[cluster] = current_vel
        clusters_best_particle[cluster] = current_best_particle
        clusters_best_path[cluster] = current_best_path

    print("Ended PSO on each cluster")
    # Now update overall params
    inertia_weight = original_inertia_weight
    results_for_each_iteration = [[t, best_overall_distance]]
    for t in range(1, max_number_of_iterations + 1):

        if t % 100 == 0:
            print("Iteration_overall: {}, min_distance: {}".format(t, best_overall_distance))
        # Update overall particles

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

        results_for_each_iteration.append([t, best_overall_distance])
        if hybrid_evolutionary_approach:
            best_elements_indexes = np.argsort(final_complete_distances)[0:int(ro / 2)]
            worst_elements_indexes = np.argsort(final_complete_distances)[int(ro / 2):]
            overall_pop[worst_elements_indexes] = overall_pop[best_elements_indexes]
            overall_vel[worst_elements_indexes] = overall_vel[worst_elements_indexes]

        if t > best_overall_iteration + wait_interval:
            print("The algorithm is no longer improving")
            break

        inertia_weight = inertia_weight * decrement_factor
        if inertia_weight < 0.4:
            inertia_weight = 0.4
        overall_vel = inertia_weight * overall_vel

    return results_for_each_iteration, final_path, clusters_best_path
