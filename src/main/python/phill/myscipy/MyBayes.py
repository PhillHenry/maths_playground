import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

if __name__ == "__main__":
    dist = stats.beta
    x = np.linspace(0, 1, 100)

    b_2_5 = dist.pdf(x, 2, 5)
    b_65_42 = dist.pdf(x, 65, 42)

    plt.plot(x, b_2_5, label="Prior [beta(2, 5)]", c='r')
    plt.plot(x, b_65_42, label="Posterior [beta(65, 42)]", c='b')
    plt.title("Bias and Posterior")
    plt.legend(loc='upper left')

    plt.show()
