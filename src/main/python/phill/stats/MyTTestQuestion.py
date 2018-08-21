import scipy.stats as stats
import numpy as np

if __name__ == '__main__':
    Nsamp = 10 ** 6
    sigma_x = 2
    mean = 100
    ps = []
    for x in range(10):
        x = np.random.normal(mean, sigma_x, size=(Nsamp))
        y = np.random.normal(mean, sigma_x, size=(Nsamp))
        (t_value, p_value) = stats.ttest_ind(x, y, equal_var=True)
        print("t-test compared to same distribution but different data, t = {0:.3f}, p = {1:.3f}".format(t_value, p_value))
        ps.append(p_value)
    p_mean = sum(ps) / len(ps)
    print("p-values: Average {0:.3f} lowest {1:.3f}".format(p_mean, min(ps)))
