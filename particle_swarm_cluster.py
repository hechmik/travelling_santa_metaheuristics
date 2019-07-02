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


def create_pop_velocities(pop_size, particle_length):
    pop_vel = [create_particle_velocities(particle_length) for i in range(0, pop_size)]
    return np.array(pop_vel)


def generate_santas_path_from_particles_pop(population):
    path_populations = np.argsort(population)
    return path_populations


def evaluate_paths_cluster(paths, cities, not_primes_bool):
    distances = []
    for path in paths:
        distances.append(santas_path.total_length(path, cities))
    return np.array(distances)


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
        # print(distances)
        best_path_index = np.argsort(distances)[0]
        # print(best_path_index)
        best_path = paths[best_path_index]
        # print(best_path)
        clusters_best_path.append(best_path)

        clusters_best_particle.append(pop[best_path_index])
    return clusters_pop, clusters_vel, clusters_best_path, clusters_best_particle, clusters_distances


def compute_final_paths(paths, clusters_els):
    end_paths = []
    clusters_els = np.array(clusters_els)
    for path in paths:

        current_path = []
        path = np.array(path)
        for cluster_path in clusters_els[path]:
            current_path = np.concatenate((current_path, cluster_path))
        #current_path = np.delete(current_path, np.argwhere(current_path == 0))
        current_path = current_path.astype(int)
        end_paths.append(current_path)
    return end_paths


def cluster_particle_swarm_optimization(
        cities,
        clusterized_cities,
        ro=30,
        number_of_iterations=100,
        decrement_factor=0.975,
        c1=2,  # cognitive param
        c2=2,  # social param
        inertia_weight=0.99,
        wait_interval=40):
    #clusterized_cities = clusterized_cities[1:]
    np_not_prime = np.vectorize(santas_path.not_prime)
    nums = np.arange(0, len(cities))
    not_primes_bool = np_not_prime(nums)
    t = 0

    cluster_dict = {}
    for cluster_index in set(clusterized_cities):
        cluster_dict[cluster_index] = np.where(clusterized_cities == cluster_index)[0].shape[0]
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
    number_of_clusters = len(clusters_pop)
    overall_pop = create_pop_particles(pop_size=ro, particle_length=number_of_clusters)

    overall_vel = create_pop_velocities(pop_size=ro, particle_length=number_of_clusters)

    overall_paths = generate_santas_path_from_particles_pop(overall_pop)
    final_complete_paths = compute_final_paths(overall_paths, clusters_best_path)

    final_complete_distances = evaluate_paths(paths=final_complete_paths, cities=cities,
                                              not_primes_bool=not_primes_bool)
    best_final_complete_distances = final_complete_distances
    best_overall_pop = overall_pop
    best_overall_particle = overall_pop[np.argsort(final_complete_distances)[0]]
    final_path = final_complete_paths[np.argsort(final_complete_distances)[0]]
    best_overall_iteration = t
    best_overall_distance = np.sort(final_complete_distances)[0]

    for t in range(1, number_of_iterations):
        if t % 20 == 0:
            print("Iteration: {}, min_distance: {}".format(t, best_overall_distance))
        inertia_weight = inertia_weight * decrement_factor
        if inertia_weight < 0.4:
            inertia_weight = 0.4
        # Improve clusters particles
        for cluster in range(number_of_clusters):
            current_pop = clusters_pop[cluster]
            current_best_pop = clusters_best_pop[cluster]
            current_vel = clusters_vel[cluster]
            current_best_particle = clusters_best_particle[cluster]

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
                #print("Cluster: {}, distance: {}".format(cluster, local_best_particle_distance))
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
            best_overall_distance = current_min_distance
            final_path = final_complete_paths[current_iteration_best_index]
    return best_overall_iteration, best_overall_distance, final_path


def cluster_particle_swarm_optimization_bottom_up(
        cities,
        clusterized_cities,
        ro=30,
        number_of_iterations=100,
        decrement_factor=0.975,
        c1=2,  # cognitive param
        c2=2,  # social param
        inertia_weight=0.99,
        wait_interval=40):

    np_not_prime = np.vectorize(santas_path.not_prime)
    nums = np.arange(0, len(cities))
    not_primes_bool = np_not_prime(nums)
    t = 0

    cluster_dict = {}
    for cluster_index in set(clusterized_cities):
        cluster_dict[cluster_index] = np.where(clusterized_cities == cluster_index)[0].shape[0]
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
    number_of_clusters = len(clusters_pop)
    overall_pop = create_pop_particles(pop_size=ro, particle_length=number_of_clusters)

    overall_vel = create_pop_velocities(pop_size=ro, particle_length=number_of_clusters)

    overall_paths = generate_santas_path_from_particles_pop(overall_pop)
    final_complete_paths = compute_final_paths(overall_paths, clusters_best_path)
    final_complete_distances = evaluate_paths(paths=final_complete_paths, cities=cities,
                                              not_primes_bool=not_primes_bool)
    best_final_complete_distances = final_complete_distances
    best_overall_pop = overall_pop
    best_overall_particle = overall_pop[np.argsort(final_complete_distances)[0]]
    final_path = final_complete_paths[np.argsort(final_complete_distances)[0]]
    best_overall_iteration = t
    best_overall_distance = np.sort(final_complete_distances)[0]

    for t in range(1, number_of_iterations + 1):
        if t % 20 == 0:
            print("Iteration_Cluster: {}".format(t))
        inertia_weight = inertia_weight * decrement_factor
        if inertia_weight < 0.4:
            inertia_weight = 0.4
        # Improve clusters particles
        for cluster in range(number_of_clusters):
            current_pop = clusters_pop[cluster]
            current_best_pop = clusters_best_pop[cluster]
            current_vel = clusters_vel[cluster]
            current_best_particle = clusters_best_particle[cluster]

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
                #print("Cluster: {}, distance: {}".format(cluster, local_best_particle_distance))

     # Now update overall params

    for t in range(1, number_of_iterations + 1):
        if t % 20 == 0:
            print("Iteration_overall: {}, min_distance: {}".format(t, best_overall_distance))
        # Update overall particles
        overall_vel = overall_vel + c1 * r1 * (best_overall_pop - overall_pop)
        overall_vel = overall_vel + c2 * r2 * (best_overall_particle - overall_pop)
        overall_pop = overall_pop + overall_vel

        overall_paths = generate_santas_path_from_particles_pop(overall_pop)
        # print(overall_paths)
        final_complete_paths = compute_final_paths(overall_paths, clusters_best_path)
        # print(final_complete_paths)
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
            # print(best_overall_particle)
            final_path = final_complete_paths[current_iteration_best_index]
            best_overall_distance = current_min_distance

    return best_overall_iteration, best_overall_distance, final_path
