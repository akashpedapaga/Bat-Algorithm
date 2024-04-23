import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Define fitness function
def Fitness(L, X_train, y_train, classifier):
    # Assuming L represents a set of features selected by Dolphin Echolocation
    # Train a classifier and evaluate its performance
    selected_features = [X_train.columns[i] for i in L]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    classifier.fit(X_train_selected, y_train)
    return classifier.score(X_test_selected, y_test)

# Define the Dolphin Echolocation Algorithm function
def DolphinEcholocation(X_train, y_train, NL, NV, LoopsNumber, Re, Epsilon, multiplier, classifier):
    xmin = 0  # Adjust xmin and xmax based on the number of features in your dataset
    xmax = len(X_train.columns) - 1
    PP = []
    PP1 = 0.11
    Power = -0.5
    L = []
    Fit = []
    Alternatives = []
    AF = []

    # Generate initial alternatives and initialize fitness matrices
    for i in range(xmin, xmax + 1):
        a = []
        af = []
        for j in range(NV):
            a.append(i)
            af.append(0)
        Alternatives.append(a)
        AF.append(af)

    # Generate initial random locations and fitnesses
    for i in range(NL):
        li = []
        for j in range(NV):
            li.append(int(random.uniform(xmin, xmax)))
        L.append(li)
        Fit.append(0)

    # Main loop
    for loop in range(1, LoopsNumber):
        PPi = PP1 + ((1 - PP1) * ((loop ** Power) - 1) / ((LoopsNumber ** Power) - 1))
        PP.append(PPi)

        # Evaluate fitness of each location
        for LocationNumber in range(NL):
            Fit[LocationNumber] = Fitness(L[LocationNumber], X_train, y_train, classifier)

        # Update Accumulative Fitness matrix
        for i in range(NL):
            for j in range(NV):
                jthColumnOfAlt = column(Alternatives, j)
                A = jthColumnOfAlt.index(L[i][j])
                for k in range(-Re, Re):
                    if (A + k) >= 0 and (A + k) < len(Alternatives):
                        s = A + k
                        AF[A + k][j] += (1 / Re) * (Re - abs(k)) * Fit[i]

        # Add epsilon
        AF = addEpsilon(AF, Epsilon)

        # Update probabilities
        BestLocation = L[Fit.index(max(Fit))]
        for j in range(NV):
            for i in range(len(Alternatives)):
                if Alternatives[i][j] == BestLocation[j]:
                    AF[i][j] = 0

        sumAF = 0
        for i in range(len(AF)):
            for j in range(len(AF[0])):
                sumAF += AF[i][j]
        P = []
        for i in range(len(Alternatives)):
            p = []
            for j in range(NV):
                p.append(AF[i][j] / sumAF)
            P.append(p)

        for j in range(NV):
            for i in range(len(Alternatives)):
                if Alternatives[i][j] == BestLocation[j]:
                    P[i][j] = PPi
                else:
                    P[i][j] *= (1 - PPi)

        # Generate next L matrix according to probabilities
        pool = []
        for j in range(NV):
            temp = []
            for i in range(len(Alternatives)):
                for t in range(int(P[i][j] * multiplier)):
                    temp.append(Alternatives[i][j])
            pool.append(temp)
        for i in range(NL):
            for j in range(NV):
                L[i][j] = random.choice(pool[j])

        # Scatter plot of search space exploration
        x = [i[0] for i in L]
        y = [i[1] for i in L]
        plt.scatter(x, y)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Search Space Exploration')
        plt.show()

    return BestLocation

# Function to extract jth column from a given matrix A
def column(A, j):
    A_j = []
    for i in range(len(A)):
        A_j.append(A[i][j])
    return A_j

# Add the value e to every element of the given matrix A
def addEpsilon(A, e):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] += e
    return A

# Load dataset
data = pd.read_csv('F:\Local Disk F\MS CS\Academics\Projects\AIProj\Dataset\diabetes.csv')

# Separate features and target variable
X = data.drop('Outcome', axis=1) 
y = data['Outcome'] 

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)  # Convert X_train_scaled back to DataFrame

# Define DNN classifier
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

# Run Dolphin Echolocation Algorithm
best_features = DolphinEcholocation(X_train, y_train, NL=50, NV=2, LoopsNumber=8, Re=10, Epsilon=0.01, multiplier=10000, classifier=classifier)

# Print the best location
print("Best Location:", best_features)

# Evaluate performance
selected_features = [X_train.columns[i] for i in best_features]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
classifier.fit(X_train_selected, y_train)
y_pred = classifier.predict(X_test_selected)

# Calculate metrics
precision = precision_score(y_test, y_pred, zero_division=1.0) 
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)