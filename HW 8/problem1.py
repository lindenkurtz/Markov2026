import numpy as np
from scipy.special import gamma, factorial
from scipy.stats import poisson

def p_k(k, a, b, t):
    term1 = gamma(a + k) / (gamma(a) * factorial(k))
    term2 = (b / (b + t)) ** a
    term3 = (t / (b + t)) ** k
    return term1 * term2 * term3

prob_0_9 = sum(p_k(k, 19, 12, 5) for k in range(10))
print(f'prob = {1 - prob_0_9:.4f}')

print(f'prob2 = {1 - poisson.cdf(9, 5 * 3 / 2):.4f}')