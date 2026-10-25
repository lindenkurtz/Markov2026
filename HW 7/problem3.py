import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(5)

def exp_sim(total_time, lam):
    T_arr = []
    T = 0
    while True:
        t = rng.exponential(1/lam)
        T += t
        if T >= total_time:
            break
        T_arr.append(T)
    return T_arr

def unif_sim(total_time, lam):
    N = rng.poisson(lam * total_time)
    t_arr = rng.uniform(0, total_time, N)
    T_arr = np.sort(t_arr)
    return T_arr

# simulation
lam = 4
hours = int(2e5)
sims = {"exponential gaps": np.array(exp_sim(hours, lam)), "poisson + uniform": unif_sim(hours, lam)}

for name, T in sims.items():
    N1 = np.bincount(T.astype(int), minlength=hours)
    halves = np.bincount((2 * T).astype(int), minlength=2 * hours).reshape(hours, 2)
    first = np.searchsorted(T, np.arange(hours))
    T1 = T[first[:-1]] - np.arange(hours - 1)
    T3 = T[first[:-3] + 2] - np.arange(hours - 3)
    three = N1[:-1] == 3
    print(f"\n{name}:")
    print(f"  mean N(1) = {N1.mean():.4f}, var N(1) = {N1.var():.4f}  (exact 4, 4)")
    print(f"  P(N(1/2)=0 | N(1)=3) = {np.mean(halves[:-1][three, 0] == 0):.4f}  (exact 0.125)")
    print(f"  E[T1 | N(1)=3] = {T1[three].mean():.4f}  (exact 0.25)")
    print(f"  corr(first half, second half) = {np.corrcoef(halves[:, 0], halves[:, 1])[0, 1]:.4f}  (exact 0)")

    # N(1) histogram vs Poisson(4)
    k = np.arange(N1.max() + 1)
    plt.bar(k, np.bincount(N1) / hours, alpha=0.5, label="simulated")
    plt.plot(k, stats.poisson.pmf(k, lam), "ko", label="Poisson(4) pmf")
    plt.xlabel("N(1)"); plt.ylabel("probability"); plt.title(name); plt.legend()
    plt.savefig(f"problem3_N1_{name.split()[0]}.png", dpi=150)
    plt.close()

    # T3 histogram vs Gamma(3, 4)
    t = np.linspace(0, T3.max(), 500)
    plt.hist(T3, bins=150, density=True, alpha=0.5, label="simulated")
    plt.plot(t, stats.gamma.pdf(t, 3, scale=1 / lam), "k", label="Gamma(3, 4) density")
    plt.xlabel("T3"); plt.ylabel("density"); plt.title(name); plt.legend()
    plt.savefig(f"problem3_T3_{name.split()[0]}.png", dpi=150)
    plt.close()

# rasters of first ten hours
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, T) in zip(axes, sims.items()):
    ax.eventplot([T[(T >= h) & (T < h + 1)] - h for h in range(10)])
    ax.set_title(name); ax.set_xlabel("time in hour"); ax.set_ylabel("realization")
plt.tight_layout()
plt.savefig("problem3_rasters.png", dpi=150)
plt.close()

# total variation, Binomial(t/dt, lam dt) vs Poisson(lam t)
n_arr = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
dt = 1 / n_arr
tv = []
for n in n_arr:
    k = np.arange(n + 1)
    tv.append(0.5 * (np.abs(stats.binom.pmf(k, n, lam / n) - stats.poisson.pmf(k, lam)).sum() + stats.poisson.sf(n, lam)))
slope, intercept = np.polyfit(np.log(dt), np.log(tv), 1)
print("\nTV distances:", np.round(tv, 6))
print(f"log-log slope = {slope:.4f}")

plt.loglog(dt, tv, "o", label="TV distance")
plt.loglog(dt, np.exp(intercept) * dt ** slope, "k", label=f"fit, slope {slope:.3f}")
plt.xlabel("dt"); plt.ylabel("TV distance"); plt.legend()
plt.savefig("problem3_tv.png", dpi=150)
plt.close()
