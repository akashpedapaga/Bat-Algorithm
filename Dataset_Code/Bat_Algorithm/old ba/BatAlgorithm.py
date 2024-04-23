import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def initialize_bats(pop_size, dim):
    return np.random.rand(pop_size, dim)

def update_position(position, velocity):
    return position + velocity

def bat_algorithm(objective_function, pop_size=10, max_iterations=100, loudness=0.5, pulse_rate=0.5):
    dim = objective_function.__code__.co_argcount - 1  # The objective function takes 'x' and 'y', so subtract 1 for dimensionality
    
    bats = initialize_bats(pop_size, dim)
    velocities = np.zeros((pop_size, dim))

    fitness = np.apply_along_axis(objective_function, 1, bats)
    best_index = np.argmin(fitness)
    best_solution = bats[best_index]

    for iteration in range(max_iterations):
        current_loudness = loudness * (1 - np.exp(-pulse_rate * iteration))

        for i in range(pop_size):
            frequency = 0.5
            velocities[i] = velocities[i] + (bats[i] - best_solution) * frequency
            bats[i] = update_position(bats[i], velocities[i])

            if np.random.rand() > current_loudness:
                bats[i] = best_solution + 0.001 * np.random.randn(dim)

        new_fitness = np.apply_along_axis(objective_function, 1, bats)
        new_best_index = np.argmin(new_fitness)
        
        if new_fitness[new_best_index] < fitness[best_index]:
            best_solution = bats[new_best_index]
            best_index = new_best_index

    return best_solution, fitness[best_index]

# Define the objective function
def objective_function(features):
    # Features is a binary vector indicating selected features
    selected_features = [X_train.columns[i] for i, selected in enumerate(features) if selected]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)
    
    return 1 - accuracy_score(y_test, y_pred)  # We minimize the error rate, so 1 - accuracy is used

# Load dataset
data = pd.read_csv('diabetes.csv')

# Separate features and target variable
X = data.drop('Outcome', axis=1)  # Replace 'target_column_name' with the actual target column name
y = data['Outcome']  # Replace 'target_column_name' with the actual target column name

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Size of train set:", X_train.shape[0])
print("Size of test set:", X_test.shape[0])

if X_train.empty or X_test.empty:
    raise ValueError("Empty train or test data after splitting")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Check if any NaN or infinite values exist in the data after scaling
if np.isnan(X_train).any() or np.isinf(X_train).any():
    raise ValueError("NaN or infinite values found in standardized train data")

# Define Random Forest classifier
classifier = RandomForestClassifier()

# Run bat algorithm
best_features, best_fitness = bat_algorithm(objective_function)

# Evaluate performance
selected_features = [X_train.columns[i] for i, selected in enumerate(best_features) if selected]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

missing_features = set(selected_features) - set(X.columns)
if missing_features:
    raise ValueError(f"Selected features {missing_features} are not found in the training data")

classifier.fit(X_train_selected, y_train)
y_pred = classifier.predict(X_test_selected)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

print("Selected Features:", selected_features)
print("Best Fitness (Error Rate):", best_fitness)
print("Accuracy:", accuracy)
