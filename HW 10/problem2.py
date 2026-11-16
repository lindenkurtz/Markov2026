import numpy as np
from scipy.stats import poisson
rng = np.random.default_rng(5)
import matplotlib.pyplot as plt

#part b
print(f'P(X > 8): {1 - poisson.cdf(k=8, mu=4)}')

#part c
def simulation(T, lam, mu):
    time_per_state = np.zeros(100)
    curr_state = 0
    time_elapsed = 0
    while(True):
        t = rng.exponential(1 / (lam + curr_state * mu))
        time_elapsed += t
        if time_elapsed > T:
            time_per_state[curr_state] += T - (time_elapsed - t)
            break

        time_per_state[curr_state] += t
        if rng.random() < lam / (lam + mu * curr_state):
            curr_state = curr_state + 1
        else:
            curr_state = curr_state - 1
    return time_per_state

lam, mu = 12, 3
T = 500
frac = simulation(T, lam, mu) / T
states = np.arange(15)
plt.bar(states - 0.2, frac[:15], width=0.4, label="simulated")
plt.bar(states + 0.2, poisson.pmf(states, mu=4), width=0.4, label="Poisson(4)")
plt.xlabel("state n")
plt.ylabel("fraction of time")
plt.legend()
plt.savefig("problem2.png")

diffs = np.zeros(20)
for i in range(20):
    diffs[i] = np.abs(frac[i] - poisson.pmf(i, mu=4))

print(f'Largest difference: {diffs.max()}')

print(f'Time average of X_t: {np.sum(np.arange(len(frac)) * frac)}')
print(f'Rho: 4')

tail_prob = 1
c = 0
while(tail_prob >= 0.01):
    c += 1
    tail_prob = 1 - poisson.cdf(c, mu=4)

print(f'Smallest c: {c}')

