from scipy import stats
import numpy as np

if __name__ == '__main__':
    Nsamp = 10 ** 6
    sigma_x = 2
    x = np.random.normal(0, sigma_x, size=(Nsamp))
    y = np.random.normal(0, sigma_x, size=(Nsamp))
    (t_value, p_value) = stats.ttest_ind(x, x, equal_var=True)  # t = 0.000, p = 1.000
    print("compared to self", t_value, p_value)
    shuffled = np.copy(x)
    np.random.shuffle(shuffled)
    (t_value, p_value) = stats.ttest_ind(x, shuffled, equal_var=True)  # t = 0.000, p = 1.000
    print("after shuffling", t_value, p_value)  # -3.6817806444807806e-16 0.9999999999999998
    (t_value, p_value) = stats.ttest_ind(x, y, equal_var=True)  # t = 1.204, p = 0.228
    print("compared to same distribution but different data, t = {0:.3f}, p = {1:.3f}".format(t_value, p_value))