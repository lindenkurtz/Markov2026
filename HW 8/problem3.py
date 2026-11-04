import numpy as np
from scipy.special import factorial
from scipy.stats import poisson
from scipy.optimize import minimize
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

#part a
def lam(t):
    return 12 + 8 * np.cos((np.pi * (t - 16)) / 12)

def LAM(s, t):
    return 12 * (t - s) + (96 / np.pi) * (np.sin(np.pi * (t - 16) / 12) - np.sin(np.pi * (s - 16) / 12))

def p_k(k, s, t):
    return poisson.pmf(k, LAM(s, t))

def simple_p_k(k, lam):
    return poisson.pmf(k, lam)

prob_0_10 = sum(p_k(k, 3, 4) for k in range(11))
print(f'prob detailed = {1 - prob_0_10:.4f}')
prob_0_10_simple = sum(simple_p_k(k, 12) for k in range(11))
print(f'prob simple = {1 - prob_0_10_simple:.4f}')

#part c
def sim(t):
    num_proposals = rng.poisson(t * 20)
    proposals = np.sort(rng.uniform(0, t, num_proposals))
    accepted = proposals[rng.uniform(0, 1, num_proposals) < lam(proposals) / 20]
    return accepted, num_proposals

arrivals, num_proposals = sim(720)
frac_kept = len(arrivals) / num_proposals
print(f'fraction of proposals kept = {frac_kept:.4f}')

hour = np.floor(arrivals % 24).astype(int)
day = np.floor(arrivals / 24).astype(int)

hourly_mean = np.bincount(hour, minlength=24) / 30
ts = np.linspace(0, 24, 200)
plt.bar(np.arange(24) + 0.5, hourly_mean, width=1, edgecolor='black', label='simulated mean per hour')
plt.plot(ts, lam(ts), color='red', label='true lambda(t)')
plt.xlabel('hour of day')
plt.ylabel('arrivals per hour')
plt.legend()
plt.savefig('problem3c.png')

counts_3_4 = np.bincount(day[hour == 3], minlength=30)
print(f'3-4am mean: {counts_3_4.mean():.4f} var: {counts_3_4.var():.4f}  Poisson mean/var: {LAM(3, 4):.4f}')

def neg_log_lik(params, times, T):
    a, b, phi = params
    rates = a + b * np.cos(np.pi * (times - phi) / 12)
    LAM_0_T = a * T + (12 * b / np.pi) * (np.sin(np.pi * (T - phi) / 12) - np.sin(np.pi * (0 - phi) / 12))
    return -(np.sum(np.log(rates)) - LAM_0_T)

fit = minimize(neg_log_lik, [10, 5, 14], args=(arrivals, 720), method='Nelder-Mead')
a_hat, b_hat, phi_hat = fit.x
print(f'MLE: a = {a_hat:.4f} b = {b_hat:.4f} phi = {phi_hat:.4f}')

n = len(arrivals)
log_lik_const = n * np.log(n / 720) - n
print(f'log-lik cosine: {-fit.fun:.4f}  log-lik constant: {log_lik_const:.4f}  difference: {-fit.fun - log_lik_const:.4f}')
