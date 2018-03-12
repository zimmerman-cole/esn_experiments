import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def run(filename):
    a = pkl.load(open(filename, "rb"))
    plt.plot(range(len(a[0])), a[0], label="pop")
    plt.plot(range(len(a[1])), a[1], label="best")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run("ESN_GA_SIGMOID__DATApartial.pkl")