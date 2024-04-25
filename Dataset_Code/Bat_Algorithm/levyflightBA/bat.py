import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import pca

class Bat:
    def __init__(self, d, pop, numOfGenerations, a, r, q_min, q_max, lower_bound, upper_bound, function, use_pca=True, levy=False, seed=0, alpha=1, gamma=1):
        # Number of dimensions
        self.d = d
        # Population size
        self.pop = pop
        # Generations
        self.numOfGenerations = numOfGenerations
        # Loudness and alpha parameter (0 < a < 1)
        self.A = np.array([a] * pop)
        self.alpha = alpha
        # Pulse rate and gamma parameter (y > 0)
        self.R = np.array([r] * pop)
        self.gamma = gamma
        # (Min/Max) frequency
        self.Q = np.zeros(self.pop)
        self.q_min = q_min
        self.q_max = q_max
        # Domain bounds
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.levy = levy
        self.use_pca = use_pca
        if use_pca:
            self.PCA = pca.PCA()

        # Initialise fitness and solutions
        self.f_min = np.inf
        self.solutions = np.zeros((self.pop, self.d))
        self.pop_fitness = np.zeros(self.pop)  # fitness of population
        self.best = np.zeros(self.d)  # best solution

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Velocity
        self.V = np.zeros((self.pop, self.d))

        # Optimisation/fitness function
        self.func = function

        # History (for plots)
        self.best_history = []
        self.min_val_history = []
        self.loudness_history = []
        self.pulse_rate_history = []
        self.frequency_history = []


    def find_best_bat(self):
        j = 0
        for i in range(self.pop):
            if self.pop_fitness[i] < self.pop_fitness[j]:
                j = i
        for i in range(self.d):
            self.best[i] = self.solutions[j][i]
        self.f_min = self.pop_fitness[j]

    def init_bats(self):
        for i in range(self.pop):
            for j in range(self.d):
                self.solutions[i][j] = self.lower_bound + (self.upper_bound - self.lower_bound) * self.rng.uniform(0, 1)
            self.pop_fitness[i] = self.func(self.solutions[i])

        self.find_best_bat()

    def update_frequency(self, i):
        self.Q[i] = self.q_min + (self.q_max - self.q_min) * self.rng.uniform(0, 1)

    def update_loudness(self, i):
        self.A[i] *= self.alpha

    def update_pulse_rate(self, i, t):
        self.R[i] = self.gamma * (1 - np.exp(-self.gamma * t))

    def global_search(self, X, i):
        """ Update velocity and location based on Eq.3 and Eq.4 in [1] """
        for j in range(self.d):
            self.V[i][j] = self.V[i][j] + (self.solutions[i][j] - self.best[j]) * self.Q[i]
            X[i][j] = np.clip(self.solutions[i][j] + self.V[i][j],
                self.lower_bound, self.upper_bound)

    def levy_rejection(self):
        x = self.rng.uniform(0, 1)
        y = self.rng.uniform(0, 1.5) # PDF maximum peak at x=1/3 --> y~1.45
        fx = np.sqrt(1/(2*np.pi))*np.exp(-1/(2*x))/(x**1.5) # PDF
        while (y < fx):
            x = self.rng.uniform(0, 1)
            y = self.rng.uniform(0, 1.5) # PDF maximum peak at x=1/3 --> y~1.45
            fx = np.sqrt(1/(2*np.pi))*np.exp(-1/(2*x))/(x**1.5) # PDF
        return x


    def my_levy(self, u, c = 1.0, mu = 0.0):
        """    
        Arguments:
        u: a uniform[0,1) random number
        c: scale parameter for Levy distribution (defaults to 1)
        mu: location parameter (offset) for Levy (defaults to 0)
        """
        return mu + c / (2 * norm.ppf(1.0 - u)**2)

    def levy_flight(self, X, i):
        for j in range(self.d):
            X[i][j] = np.clip(self.solutions[i][j] + self.levy_rejection() * (self.solutions[i][j] - self.best[j]),
                self.lower_bound, self.upper_bound)

    def move_bats(self, X, t):
        for i in range(self.pop):

            self.update_frequency(i)

            if self.levy:
                self.levy_flight(X, i)
            else:
                self.global_search(X, i)

            if self.use_pca:
                X = self.PCA.replace_principal_individuals(X, self.lower_bound, self.upper_bound)

            # Update positions with local search if random sample is greater than pulse rate
            if self.rng.random() > self.R[i]:
                for j in range(self.d):
                    # Original bat paper
                    alpha = 0.001
                    # Optimisation, average loudness
                    alpha = np.mean(self.A)
                    X[i][j] = np.clip(self.best[j] + alpha * self.rng.normal(0, 1),
                        self.lower_bound, self.upper_bound)

            f_new = self.func(X[i])


            if (f_new < self.pop_fitness[i]) and (self.rng.random() < self.A[i]):
                for j in range(self.d):
                    self.solutions[i][j] = X[i][j]
                self.pop_fitness[i] = f_new
                self.update_loudness(i)
                self.update_pulse_rate(i, t)

            # Re-evaluate best bat
            if f_new < self.f_min:
                for j in range(self.d):
                    self.best[j] = X[i][j]
                self.f_min = f_new

    def keep_history(self):
        self.best_history.append(self.best)
        self.min_val_history.append(self.f_min)
        self.loudness_history.append(np.mean(self.A))
        self.pulse_rate_history.append(np.mean(self.R))
        self.frequency_history.append(np.mean(self.Q))


    def run_bats(self):
        """ Actual run function, returns dictionary with results. """
        X = np.zeros((self.pop, self.d))

        self.init_bats()

        for t in tqdm(range(self.numOfGenerations)):
            self.move_bats(X, t)
            self.keep_history()

        history = {}
        history.update({
            "best" : self.best_history,
            "min_val" : self.min_val_history,
            "loudness" : self.loudness_history,
            "pulserate" : self.pulse_rate_history,
            "frequency" : self.frequency_history
            })

        d = {}
        d.update({
            "best" : self.best,
            "final_fitness" : self.f_min,
            "history" : history,
            })

        return d
