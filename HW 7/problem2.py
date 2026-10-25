import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

lam = np.array((3, 2, 1.5, 1.2, 1, 0.75))

def expected_recursive(s):
    if (len(s) == 0): return 0
    total = 0
    for i in range(len(s)):
        p_next = s[i] / np.sum(s)
        sub = expected_recursive(np.delete(s, i))
        total += p_next * sub
    return 1 / np.sum(s) + total

def last_recursive(s):
    if (len(s) == 1): return np.array([1.0])
    total = np.zeros(len(s))
    for i in range(len(s)):
        p_next = s[i] / np.sum(s) 
        sub = last_recursive(np.delete(s, i)) 
        total += p_next * np.insert(sub, i, 0.0)
    return total

def expected_alternating(s):
    total = 0.0
    for r in range(1, len(s) + 1):
        sign = (-1) ** (r + 1)
        for group in combinations(s, r):
            total += sign / sum(group)
    return total

# exact values
print("E[M] recursive:  ", expected_recursive(lam))
print("E[M] alternating:", expected_alternating(lam))
print("P(last) exact:   ", np.round(last_recursive(lam), 4))

# simulation
rng = np.random.default_rng(0)
N = 200_000
T = rng.exponential(1 / lam, size=(N, len(lam)))
M = T.max(axis=1)
print("E[M] simulated:  ", M.mean())
print("P(last) sim:     ", np.round(np.bincount(T.argmax(axis=1), minlength=len(lam)) / N, 4))

# histogram
t = np.linspace(0, M.max(), 500)
F = 1 - np.exp(-np.outer(t, lam))
f = sum(lam[i] * np.exp(-lam[i] * t) * np.prod(np.delete(F, i, axis=1), axis=1) for i in range(len(lam)))
plt.hist(M, bins=150, density=True, alpha=0.5, label="simulated")
plt.plot(t, f, "k", label="exact density")
plt.xlabel("M"); plt.ylabel("density"); plt.legend()
plt.savefig("problem2_1.png", dpi=150)
plt.close()

# first finishing time
first, who = T.min(axis=1), T.argmin(axis=1)
print("1/sum(lam):", 1 / lam.sum())
for i in range(len(lam)):
    print(f"student {i+1}: P(first) = {np.mean(who == i):.4f} (exact {lam[i]/lam.sum():.4f}), "
          f"E[first | i first] = {first[who == i].mean():.4f}")

# rates
ks = range(1, 13)
EM = [expected_alternating(np.arange(1, k + 1)) for k in ks]
for k, e in zip(ks, EM):
    print(f"k = {k:2d}: E[M] = {e:.4f}")
plt.plot(ks, EM, "o-", label="E[M]")
plt.axhline(2 * np.log(2), color="r", ls="--", label="2 ln 2")
plt.xlabel("k"); plt.ylabel("E[M]"); plt.legend()
plt.savefig("problem2_2.png", dpi=150)
plt.close()
