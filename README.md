# Metaheuristics - Travellling Santa

# Problem description
You are provided a list of cities and their coordinates in cities.csv. You must create the shortest possible path that visits all the cities. Your submission file is simply the ordered list in which you visit each city. Paths have the following constraints:

Paths must start and end at the North Pole (CityId = 0)
You must visit every city exactly once
The distance between two paths is the 2D Euclidean distance, except...
Every 10th step (stepNumber % 10 == 0) is 10% more lengthy unless coming from a prime CityId.

# Algorithms choices:
For solving this problem, we have chosen to use the following algorithms:
- Genetic Algorithm
- Particle Swarm Optimization
