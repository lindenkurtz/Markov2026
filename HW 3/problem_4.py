import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

rng = np.random.default_rng(5)

# Calculations

def one_lion(R, N):
    lamb_pos = np.zeros(R)
    lamb_caught_step = np.zeros(R)
    lion_pos = np.zeros(R) + 10
    for i in range(N):
        U_lamb = rng.random(R)
        U_lion = rng.random(R)
        lamb_pos = lamb_pos + (U_lamb < 0.5) - (U_lamb >= 0.5)
        lion_pos = lion_pos + (U_lion < 0.5) - (U_lion >= 0.5)
        newly_caught = (lamb_pos == lion_pos) & (lamb_caught_step == 0)
        lamb_caught_step[newly_caught] = i + 1
    return lamb_caught_step # uncaught lambs have value 0

def two_lions(R, N):
    lamb_pos = np.zeros(R)
    lamb_caught_step = np.zeros(R)
    lion_pos = np.zeros((R, 2)) + 10
    for i in range(N):
        U_lamb = rng.random(R)
        U_lion = rng.random((R, 2))
        lamb_pos = lamb_pos + (U_lamb < 0.5) - (U_lamb >= 0.5)
        lion_pos = lion_pos + (U_lion < 0.5) - (U_lion >= 0.5)
        newly_caught = (lamb_pos[:, None] == lion_pos).any(axis=1) & (lamb_caught_step == 0)
        lamb_caught_step[newly_caught] = i + 1
    return lamb_caught_step # uncaught lambs have value 0

#plot
R, N, d0 = 2 * 10**4, 10**4, 10
t = np.arange(1, N + 1)

def survival(caught):
    counts = np.bincount(caught[caught > 0].astype(int), minlength=N + 1)
    return (1 - np.cumsum(counts) / R)[1:]

S1, S2 = survival(one_lion(R, N)), survival(two_lions(R, N))

mask = (t >= 10**2) & (t <= 10**4)
fit = lambda S: np.polyfit(np.log(t[mask]), np.log(S[mask]), 1)
beta1, logA1 = fit(S1)
beta2, logA2 = fit(S2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(t, S1, lw=1, label=r"$S_1(t)$")
ax.loglog(t, S2, lw=1, label=r"$S_2(t)$")
ax.loglog(t, S1**2, lw=1, label=r"$S_1(t)^2$")
ax.loglog(t[mask], np.exp(logA1) * t[mask]**beta1, "k--", lw=1,
          label=rf"fit $\beta_1 = {-beta1:.3f}$")
ax.loglog(t[mask], np.exp(logA2) * t[mask]**beta2, "k-.", lw=1,
          label=rf"fit $\beta_2 = {-beta2:.3f}$")
ax.loglog(t, erf(d0 / (2 * np.sqrt(t))), ":", lw=1,
          label=r"$\mathrm{erf}\!\left(d_0/(2\sqrt{t})\right)$")

ax.set_xlabel(r"$t$")
ax.set_ylabel("survival probability")
ax.set_title(rf"$\hat\beta_1 = {-beta1:.3f}$, $\hat\beta_2 = {-beta2:.3f}$ "
             rf"(fit over $10^2 \leq t \leq 10^4$)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.savefig("problem_4.png", dpi=150)
plt.show()

for tt in (10**2, 10**3, 10**4):
    i = tt - 1
    print(f"t = {tt:>6}   S2 = {S2[i]:.4f}   S1^2 = {S1[i]**2:.4f}")