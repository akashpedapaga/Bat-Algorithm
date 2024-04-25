import matplotlib.pyplot as plt

def plot_history(history, gens, name):
    figure = plt.figure()
    plt.plot(list(range(gens)), history)
    plt.savefig(f'{name}.png')
