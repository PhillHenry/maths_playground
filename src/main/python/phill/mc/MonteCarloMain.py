import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

l1 = [0.9480880153813288, 0.9508302781971102, 0.9526527161276686,
      0.951900861163621, 0.9492505353319057, 0.9510564898663217,
      0.9427974947807933, 0.9433445661331087, 0.9472660471147611]
l2 = [0.9351491569390402, 0.9421977556637731, 0.9387062566277836,
      0.9475262368815592, 0.9456567352077213, 0.9437367303609342,
      0.947907949790795, 0.9440574203082119, 0.9464398235664776]

l1_mean_obs = np.mean(l1)
l2_mean_obs = np.mean(l2)

l1_std_obs = np.std(l1, ddof=1)
l2_std_obs = np.std(l2, ddof=1)

print("L1: mean {0:.3f}, std {1:.4f} ".format(l1_mean_obs, l1_std_obs))
print("L2: mean {0:.3f}, std {1:.4f} ".format(l2_mean_obs, l2_std_obs))

# Initially, this gave me ""must use protocol 4 or greater to copy this object""
# but it was cured with:
# sudo pip3 install -U git+https://github.com/pymc-devs/pymc3.git
if __name__ == '__main__':

    # See chapter 2 of Bayesian Methods for Hackers
    with pm.Model() as model:
        l1_mean = pm.Uniform("l1_mean", 0.9, 0.99)
        l2_mean = pm.Uniform("l2_mean", 0.9, 0.99)

        l1_std = pm.Uniform("l1_std", 0.0, 0.02)
        l2_std = pm.Uniform("l2_std", 0.0, 0.02)

        l1_norm = pm.Normal("l1", l1_mean, l1_std, observed=l1)
        l2_norm = pm.Normal("l2", l2_mean, l2_std, observed=l2)

        delta = pm.Deterministic("delta", l1_mean - l2_mean)

        step = pm.Metropolis()
        trace = pm.sample(20000, step=step)
        burned_trace = trace[1000:]

        delta_samples = burned_trace["delta"]

    plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of delta", color="#7A68A6", normed=True)

    plt.show()
