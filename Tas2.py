import numpy as np
from OptimizationTestFunctions import Sphere
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def objective_function(x):
    dim = len(x) if len(x) > 0 else 1  # Ensure dimension is at least 1
    return Sphere(dim)(x)

def initialize_bats(pop_size, dim):
    return np.random.rand(pop_size, dim)

def update_position(position, velocity):
    return position + velocity

def bat_algorithm(objective_function, X_train, X_test, y_train, y_test, pop_size=10, max_iterations=100, loudness=0.5, pulse_rate=0.5):
    # Get dimensionality from the objective function
    dim = X_train.shape[1]

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

    # Evaluate the best solution on the test set
    y_pred = np.round(best_solution.dot(X_test.T))
    y_pred = np.where(y_pred >= 0.5, 1, 0)  # Convert probabilities to binary predictions
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return best_solution, accuracy, precision, recall, f1

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Assume 'Outcome' is the target column
# For demonstration, we assume no preprocessing is needed

# Split the data into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage
pop_size = 20
max_iterations = 100
loudness = 0.5
pulse_rate = 0.5

# Run the bat algorithm with the provided objective function
best_solution, accuracy, precision, recall, f1 = bat_algorithm(objective_function, X_train, X_test, y_train, y_test, pop_size, max_iterations, loudness, pulse_rate)

print("Best Solution:", best_solution)
print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
