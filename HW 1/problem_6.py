import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=5)

def mc_estimate(n):
    X = rng.random(n)
    Y = rng.random(n)
    Z = rng.random(n)

    sum = np.sum((np.square(X) + np.square(Y) < Z) & (np.square(Z) > X * Y))

    return sum / n

x = np.geomspace(10, 10**7, 60).astype(int)
y = np.array([mc_estimate(n) for n in x])
analytic = 23 * np.pi / 192

fig, ax = plt.subplots()
ax.plot(x, y, label="Monte Carlo estimate")
ax.axhline(analytic, color="red", linestyle="--",  label="Analytic value")
ax.set_xscale("log")
ax.set_xlabel("sample size $N$")
ax.set_ylabel("estimated probability")
ax.legend()

fig.savefig("problem_6.png", dpi=200, bbox_inches="tight")
plt.show()