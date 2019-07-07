# Metaheuristics - Travellling Santa

# Problem description
You are provided a list of cities and their coordinates in cities.csv. You must create the shortest possible path that visits all the cities. Your submission file is simply the ordered list in which you visit each city. Paths have the following constraints:

Paths must start and end at the North Pole (CityId = 0)
You must visit every city exactly once
The distance between two paths is the 2D Euclidean distance, except...
Every 10th step (stepNumber % 10 == 0) is 10% more lengthy unless coming from a prime CityId.

# Algorithms choices
For solving this problem, the following algorithms were implemented:
- Genetic Algorithm;
- Particle Swarm Optimization;
- Simulated Annealing.

# Methodology
The following steps were adopted to solve this problem:
- Reduction of search space through K-means clustering;
- Finding optimal subroutes inside each cluster;
- Finding an optimal sequence of clusters;
- Using SA with a mutation function that deals with non-prime 10th steps to fine-tune the routes.
A more in-depth description of the steps con be found in the report.

# Running the code
- santas_path.py contains the main distance functions used throughout the project
- ga.py contains the functions needed to run the Genetic Algorithm
- particle_swarm.py contains the functions needed to run Particle Swarm Optimiazion
- sa.py contains the functions needed to run Simulated Annealing
- full_steps_GA.ipynb is the notebook with the full execution of the main steps using GA
- PSO_with_Clustering.ipynb is the notebook with the full execution of the main steps using PSO
