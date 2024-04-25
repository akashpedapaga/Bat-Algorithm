import numpy as np
import scipy


def willem(x):
    return x / x / x / x


# Rosenbrock test function
def FRosenbrock(x):
    return scipy.optimize.rosen(x)


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def stub(x):
    value = 0.0
    for i in range(x.shape[0]):
        value += np.sum((-x[i] * np.sin(np.sqrt(np.abs(x[i])))))
    return value


# Sphere test function
def FSphere(x):
    """
    Sphere function
    range: [np.NINF, np.inf]
    """
    return (x**2).sum()


# Rastrigin test function
def FRastrigin(x):
    """
    Rastrigin's function
    [-5.12, 5.12]
    """
    return np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10)


# Griewank test function
def FGrienwank(x):
    """Griewank's function"""
    i = np.sqrt(np.arange(x.shape[0]) + 1.0)
    return np.sum(x**2) / 4000.0 - np.prod(np.cos(x / i)) + 1.0


# Weierstrass test function
def FWeierstrass(x):
    """Weierstrass's function"""
    alpha = 0.5
    beta = 3.0
    kmax = 20
    D = x.shape[0]
    exprf = 0.0

    c1 = alpha ** np.arange(kmax + 1)
    c2 = 2.0 * np.pi * beta ** np.arange(kmax + 1)
    f = 0
    c = -D * np.sum(c1 * np.cos(c2 * 0.5))

    for i in range(D):
        f += np.sum(c1 * np.cos(c2 * (x[i] + 0.5)))
    return f + c


def F8F2(x):
    f2 = 100.0 * (x[0] ** 2 - x[1]) ** 2 + (1.0 - x[0]) ** 2
    return 1.0 + (f2**2) / 4000.0 - np.cos(f2)


# FEF8F2 function
def FEF8F2(x):
    D = x.shape[0]
    f = 0
    for i in range(D - 1):
        f += F8F2(x[[i, i + 1]] + 1)
    f += F8F2(x[[D - 1, 0]] + 1)
    return f
