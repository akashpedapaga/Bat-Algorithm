#!/usr/bin/env python3
import numpy as np
import plot
import testfunctions
from levybat import Bat


def run(pca, function, lb, ub, generations):
    alpha_gamma = 0.95
    algorithm = Bat(
        # Dimension
        100,
        # Population
        100,
        # Generations
        generations,
        # Loudness
        0.9,
        # Pulse rate
        0.9,
        # Min. Freq.
        0.0,
        # Max. Freq.
        5.0,
        # Lower bound
        lb,
        # Upper bound
        ub,
        function,
        alpha=0.99,
        gamma=0.9,
        use_pca=pca,
        levy=True,
    )
    return_dict = algorithm.run_bats()
    print(
        f"Best: {return_dict['best']}, values: {return_dict['final_fitness']}"
    )
    return return_dict


def main():
    # Define a run here:
    print("Willem:", testfunctions.willem(X.values))
    print("FRosenbrock:", testfunctions.FRosenbrock(X.values[0]))
    print("rosen:", testfunctions.rosen(X.values[0]))
    print("stub:", testfunctions.stub(X.values[0]))
    print("FSphere:", testfunctions.FSphere(X.values[0]))
    print("FRastrigin:", testfunctions.FRastrigin(X.values[0]))
    print("FGrienwank:", testfunctions.FGrienwank(X.values[0]))
    print("FWeierstrass:", testfunctions.FWeierstrass(X.values[0]))
    print("F8F2:", testfunctions.F8F2(X.values[0]))
    print("FEF8F2:", testfunctions.FEF8F2(X.values[0]))
    pass


if __name__ == "__main__":
    main()
