import time
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def c_value(lam):
    return -1 / (lam * (lam - 1) * np.exp(1))

# Sample
def accept_reject_sample(N, lam):
    X = np.zeros(N)
    c = c_value(lam)
    trials = 0
    for n in range(N):
        while (True):
            trials += 1
            U1 = rng.random()
            U2 = rng.random()
            X[n] = -np.log(-U1 + 1) / lam
            f = X[n] * np.exp(-X[n])
            g = (lam) * np.exp(-(lam) * X[n])
            if (U2 < f / (c * g)):
                break
    return X, trials

# Plot
def plot(N, lam):
    t0 = time.perf_counter()
    X, trials = accept_reject_sample(N, lam)
    elapsed = time.perf_counter() - t0

    c = c_value(lam)
    print(f"lam = {lam}: c = {c:.4f}, 1/c = {1/c:.4f}, "
          f"empirical = {N/trials:.4f}, {1e6*elapsed/N:.1f} us/sample\n")

    x = np.linspace(0, 5, 500)
    f = x * np.exp(-x) # true dist
    cg = c * (lam) * np.exp(-(lam) * x)

    fig, axs = plt.subplots(figsize=(8,6))
    axs.hist(X, bins='fd', density=True, label='Empirical pdf')
    axs.plot(x, f, label="True Gamma Distribution")
    axs.plot(x, cg, label="Scaled Exponential Bounding Distribution")
    axs.set_xlim([0, 5])
    axs.set_xlabel("x")
    axs.set_ylabel("f(x)")
    axs.set_title(f"lam = {lam}")
    axs.legend()
    fig.savefig(f'problem_3_lam={lam}.png', dpi=200, bbox_inches="tight")

N = 10 ** 4
for lam in [1/2, 0.2]:
    plot(N, lam)
plt.show()