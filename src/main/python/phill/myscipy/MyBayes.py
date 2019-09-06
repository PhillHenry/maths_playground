import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

if __name__ == "__main__":
    print("hello world")
    dist = stats.beta
    x = np.linspace(0, 1, 100)

    b_2_5 = dist.pdf(x, 2, 5)
    b_65_42 = dist.pdf(x, 65, 42)

    plt.plot(x, b_2_5, label="beta(2, 5)")
    plt.plot(x, b_65_42, label="beta(65, 42)")
    plt.title("Fig 9.9 from PPP")

    plt.show()
