from scipy.special import erfinv
from scipy.special import gammaincc
from scipy import optimize
from scipy.special import erfinv
import numpy as np

# see https://jakevdp.github.io/blog/2014/06/12/frequentism-and-bayesianism-3-confidence-credibility/

def approx_CI(D, sig=0.95):
    """Approximate truncated exponential confidence interval"""
    # use erfinv to convert percentage to number of sigma
    Nsigma = np.sqrt(2) * erfinv(sig)
    D = np.asarray(D)
    N = D.size
    theta_hat = np.mean(D) - 1
    return [theta_hat - Nsigma / np.sqrt(N),
            theta_hat + Nsigma / np.sqrt(N)]


def exact_CI(D, frac=0.95):
    """Exact truncated exponential confidence interval"""
    D = np.asarray(D)
    N = D.size
    theta_hat = np.mean(D) - 1

    def f(theta, D):
        z = theta_hat + 1 - theta
        return (z > 0) * z ** (N - 1) * np.exp(-N * z)

    def F(theta, D):
        return gammaincc(N, np.maximum(0, N * (theta_hat + 1 - theta))) - gammaincc(N, N * (theta_hat + 1))

    def eqns(CI, D):
        """Equations which should be equal to zero"""
        theta1, theta2 = CI
        return (F(theta2, D) - F(theta1, D) - frac,
                f(theta2, D) - f(theta1, D))

    guess = approx_CI(D, 0.68) # use 1-sigma interval as a guess
    result = optimize.root(eqns, guess, args=(D,))
    if not result.success:
        print("warning: CI result did not converge!")
    return result.x


def freq_CI_mu(D, sigma, frac=0.95):
    """Compute the confidence interval on the mean"""
    # we'll compute Nsigma from the desired percentage
    Nsigma = np.sqrt(2) * erfinv(frac)
    mu = D.mean()
    sigma_mu = sigma * D.size ** -0.5
    return mu - Nsigma * sigma_mu, mu + Nsigma * sigma_mu


def bayes_CR_mu(D, sigma, frac=0.95):
    """Compute the credible region on the mean"""
    Nsigma = np.sqrt(2) * erfinv(frac)
    mu = D.mean()
    sigma_mu = sigma * D.size ** -0.5
    return mu - Nsigma * sigma_mu, mu + Nsigma * sigma_mu


if __name__ == "__main__":
    N = 5
    Nsamp = 10 ** 6
    sigma_x = 2

    np.random.seed(0)
    x = np.random.normal(0, sigma_x, size=(Nsamp, N))
    mu_samp = x.mean(1)

    # print(np.shape(mu_samp)) # (1000000,)

    sig_samp = sigma_x * N ** -0.5

    print("{0:.3f} should equal {1:.3f}".format(np.std(mu_samp, ddof=1), sig_samp))

    true_B = 100
    sigma_x = 10

    np.random.seed(1)
    D = np.random.normal(true_B, sigma_x, size=3)
    print(D)
    print("95% Credible Region: [{0:.0f}, {1:.0f}]".format(*bayes_CR_mu(D, 10)))


    print("95% Confidence Interval: [{0:.0f}, {1:.0f}]".format(*freq_CI_mu(D, 10)))


