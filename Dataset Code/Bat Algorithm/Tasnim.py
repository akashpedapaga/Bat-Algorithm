import numpy as np
from OptimizationTestFunctions import Sphere

def objective_function(x):
    dim = len(x) if len(x) > 0 else 1  # Ensure dimension is at least 1
    return Sphere(dim)(x)

def initialize_bats(pop_size, dim):
    return np.random.rand(pop_size, dim)

def update_position(position, velocity):
    return position + velocity

def bat_algorithm(objective_function, pop_size=10, max_iterations=100, loudness=0.5, pulse_rate=0.5):
    # Get dimensionality from the objective function
    dim = objective_function.__code__.co_argcount - 1  # Subtract 1 for 'x' argument

    # Initialize bats and velocities
    bats = initialize_bats(pop_size, dim)
    velocities = np.zeros((pop_size, dim))

    # Calculate fitness for initial solutions
    fitness = np.apply_along_axis(objective_function, 1, bats)

    # Find the index of the best solution
    best_index = np.argmin(fitness)
    best_solution = bats[best_index]

    # Start the iterations
    for iteration in range(max_iterations):
        current_loudness = loudness * (1 - np.exp(-pulse_rate * iteration))

        for i in range(pop_size):
            frequency = 0.5
            velocities[i] = velocities[i] + (bats[i] - best_solution) * frequency
            bats[i] = update_position(bats[i], velocities[i])

            if np.random.rand() > current_loudness:
                bats[i] = best_solution + 0.001 * np.random.randn(dim)

        # Calculate fitness for the updated solutions
        new_fitness = np.apply_along_axis(objective_function, 1, bats)

        # Find the index of the new best solution
        new_best_index = np.argmin(new_fitness)
        if new_fitness[new_best_index] < fitness[best_index]:
            best_solution = bats[new_best_index]
            best_index = new_best_index

    return best_solution, fitness[best_index]

# Example usage
pop_size = 20
max_iterations = 100
loudness = 0.5
pulse_rate = 0.5

# Run the bat algorithm with the provided objective function
best_solution, best_fitness = bat_algorithm(objective_function, pop_size, max_iterations, loudness, pulse_rate)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
