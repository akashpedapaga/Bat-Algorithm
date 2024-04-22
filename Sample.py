import random
import matplotlib.pyplot as plt

# Define fitness function
def Fitness(L):
    h = 0
    for var in L:
        h += var * var
    return 1 / (h + 0.01)

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

# Define constants and parameters
xmin = -20
xmax = 20
NL = 50
NV = 2
LoopsNumber = 8
PP = []
PP1 = 0.11
Power = -0.5
Re = 10
Epsilon = 0.01
multiplier = 10000
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
        Fit[LocationNumber] = Fitness(L[LocationNumber])

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

    # Plot
    x = []
    y = []
    clr = []
    for pt in L:
        x.append(pt[0])
        y.append(pt[1])
        xalt = column(Alternatives, 0)
        yalt = column(Alternatives, 1)
        xidx = xalt.index(pt[0])
        yidx = yalt.index(pt[1])
        xclr = P[xidx][0]
        yclr = P[yidx][1]
        clr.append(xclr + yclr)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))
    plt.scatter(x, y, c=clr, cmap='Reds')
    plt.scatter(x, y, c='blue')
    plt.show()

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

print(BestLocation)
