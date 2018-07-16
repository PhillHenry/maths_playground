import os, sys
sys.path.insert(0, os.path.abspath("."))  # https://stackoverflow.com/questions/9427037/relative-path-not-working-even-with-init-py
import matplotlib.pyplot as plt
from scipy import stats
import MonteCarloMain as data

(t_value, p_value) = stats.ttest_ind(data.l1, data.l2, equal_var=False)
print("t = {0:.3f}, p = {1:.3f}".format(t_value, p_value))

# this doesn't quite give the correct value as the degrees of freedom is 0
vn1 = data.l1_std_obs ** 2 / len(data.l1)
vn2 = data.l2_std_obs ** 2 / len(data.l2)

t = (data.l1_mean_obs - data.l2_mean_obs) / (vn1 + vn2) ** 0.5
print("t = ", t)


if __name__ == '__main__':
    # https://stackoverflow.com/questions/23546552/1d-plot-matplotlib
    plt.hlines(1, min(min(data.l1), min(data.l2)), max(max(data.l1), max(data.l2)))  # Draw a horizontal line
    plt.eventplot(data.l1, orientation='horizontal', colors='b')
    plt.eventplot(data.l2, orientation='horizontal', colors='r')
    ax = plt.gca()  # https://stackoverflow.com/questions/3886255/how-do-i-remove-the-y-axis-from-a-pylab-generated-picture
    ax.yaxis.set_visible(False)
    plt.show()