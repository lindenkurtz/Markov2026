import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(5)

x = np.array([-1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97,
              0.79, 1.50, 2.00, 2.63, 1.19, -2.31, 0.87])

def g_log(z):
    a = z[0]
    beta = np.exp(z[1])
    return np.size(x) * np.log(beta) - np.sum(np.log(beta**2 + (x - a)**2))

def g(z):
    a = z[0]
    beta = np.exp(z[1])
    return beta ** np.size(x) * np.prod(1 / (beta ** 2 + (x - a) ** 2))

def mcmc(s, T):
    Z = np.zeros((T, 2))
    acceptances = 0
    for i in range(T - 1):
        y = rng.normal(Z[i], s)
        log_prob = min(0, g_log(y) - g_log(Z[i]))
        if (np.log(rng.uniform()) < log_prob):
            acceptances += 1
            Z[i+1] = y
        else:
            Z[i+1] = Z[i]
    return Z, acceptances / (T - 1)
        


def iat(chain):
    # 1 + 2 * sum of autocorrelations, stopping at the first negative one
    c = chain - chain.mean()
    n = len(c)
    acf = np.correlate(c, c, mode="full")[n - 1:] / (c @ c)
    tau = 1
    for k in range(1, n):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]
    return tau


T = 50000
burn = 5000
chains = {}
print("\npart c")
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for ax, s in zip(axes, [0.02, 0.5, 5]):
    Z, acc = mcmc(s, T)
    chains[s] = Z
    print(f"s = {s}: acceptance = {acc:.3f}, IAT(alpha) = {iat(Z[:, 0]):.1f}")
    ax.plot(Z[:400, 0])
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(f"s = {s}")
axes[-1].set_xlabel("step")
plt.tight_layout()
plt.savefig("problem1_c_1.png")

# best s
s_best = min(chains, key=lambda s: iat(chains[s][:, 0]))
Z = chains[s_best][burn:]
print(f"\nbest s = {s_best}")

# exact posterior on a grid over (alpha, log beta)
a_grid = np.linspace(-2, 4, 300)
b_grid = np.linspace(-1.5, 2, 300)
L = np.array([[g_log((a, b)) for a in a_grid] for b in b_grid])
P = np.exp(L - L.max())
P /= P.sum()
pa = P.sum(axis=0)
pb = P.sum(axis=1)

def grid_summary(vals, p):
    cdf = np.cumsum(p)
    return vals @ p, np.interp(0.05, cdf, vals), np.interp(0.95, cdf, vals)

def chain_summary(v):
    return v.mean(), *np.quantile(v, [0.05, 0.95])

rows = [("alpha, chain", chain_summary(Z[:, 0])),
        ("alpha, grid ", grid_summary(a_grid, pa)),
        ("beta,  chain", chain_summary(np.exp(Z[:, 1]))),
        ("beta,  grid ", grid_summary(np.exp(b_grid), pb))]
for name, (m, lo, hi) in rows:
    print(f"{name}: mean = {m:.3f}, 90% interval = ({lo:.3f}, {hi:.3f})")
print(f"sample mean = {x.mean():.3f}, sample median = {np.median(x):.3f}")

plt.figure(figsize=(6, 5))
plt.plot(Z[:, 0], Z[:, 1], ".", ms=1, alpha=0.3)
plt.contour(a_grid, b_grid, P, levels=8, colors="k")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\log\beta$")
plt.title(f"s = {s_best}")
plt.savefig("problem1_c_2.png")

# part d
print("\npart d")
reps = 2000
for n in [10, 100, 1000]:
    data = 1.5 + 0.8 * rng.standard_cauchy((reps, n))
    for name, est in [("mean  ", data.mean(axis=1)), ("median", np.median(data, axis=1))]:
        q25, q75 = np.quantile(est, [0.25, 0.75])
        print(f"n = {n:4d}, {name}: IQR/1.349 = {(q75 - q25) / 1.349:.4f}")

plt.show()
