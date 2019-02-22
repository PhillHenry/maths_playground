from scipy import stats
import numpy as np

if __name__ == '__main__':
    Nsamp = 10 ** 6
    sigma_x = 2
    mean = 100
    x = np.random.normal(mean, sigma_x, size=(Nsamp))
    y = np.random.normal(mean, sigma_x, size=(Nsamp))
    (t_value, p_value) = stats.ttest_ind(x, x, equal_var=True)  # t = 0.000, p = 1.000
    print("t-test compared to self,                                 t = {0:.3f}, p = {1:.3f}".format(t_value, p_value))
    shuffled = np.copy(x)
    np.random.shuffle(shuffled)
    (t_value, p_value) = stats.ttest_ind(x, shuffled, equal_var=True)  # t = 0.000, p = 1.000
    print("t-test after shuffling,                                  t = {0:.3f}, p = {1:.3f}".format(t_value, p_value))  # -3.6817806444807806e-16 0.9999999999999998
    (t_value, p_value) = stats.ttest_ind(x, y, equal_var=True)  # t = 1.204, p = 0.228
    print("t-test compared to same distribution but different data, t = {0:.3f}, p = {1:.3f}".format(t_value, p_value)) # t = 0.753, p = 0.451 (but p can go much lower, very random)

    print()
    (chi, p) = stats.chisquare(x, shuffled)
    print("chi-squared self with shuffle,                         t = {0:.3f}, p = {1:.3f}".format(chi, p))  # t = 11299165.837, p = 0.000
    (chi, p) = stats.chisquare(x, x)
    print("chi-squared with self,                                 t = {0:.3f}, p = {1:.3f}".format(chi, p))
    (chi, p) = stats.chisquare(x, y)
    print("chi-squared with same distribution but different data, t = {0:.3f}, p = {1:.3f}".format(chi, p))  # t = 0.000, p = 1.000

    print()
    print("kullback-leibler, same distribution but different data: {0:.3f}".format(stats.entropy(x, qk=y)))
    print("kullback-leibler, same data but shuffled              : {0:.3f}".format(stats.entropy(x, qk=shuffled)))
    print("kullback-leibler, same data                           : {0:.3f}".format(stats.entropy(x, qk=x)))
