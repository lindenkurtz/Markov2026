import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#espn projections - https://www.espn.com/college-football/story/_/id/49001728/2026-big-12-college-football-preview-predictions-top-transfers-more
espn_projections = {
    "Texas Tech": 23.1, "BYU": 15.4, "Utah": 12.0, "Kansas State": 10.3, 
    "Arizona": 10.2, "TCU": 9.0, "Houston": 8.2, "Oklahoma State": 6.7,
    "Arizona State": 6.3, "Cincinnati": 4.5, "Baylor": 4.3,  "Kansas": 3.6,
    "UCF": 2.2, "Iowa State": 1.0, "Colorado": 0.7, "West Virginia": 0.6, 

}

#data setup
df = pd.read_csv('big12.csv')
teams = sorted(set(df.home) | set(df.away))
m = len(teams)
idx = {t: k for k, t in enumerate(teams)}
df["i"] = df.home.map(idx)
df["j"] = df.away.map(idx)

g25 = df[(df.season == 2025) & (df.completed == 1)]
g26 = df[(df.season == 2026) & (df.completed == 1) & (df.week < 6)]
gpred = df[(df.season == 2026) & (df.week == 6)]

#math
def h(x):
    return 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(np.abs(2*x - 1))

def fill_matricies(games, m):
    H = np.zeros((m, m))
    C = np.zeros((m, m))
    S = games.home_pts.values
    T = games.away_pts.values
    i = games.i.values
    j = games.j.values

    hij = h((S + 1) / (S + T + 2))

    np.add.at(H, (i, j), hij)
    np.add.at(H, (j, i), 1 - hij)
    np.add.at(C, (i, j), 1)
    np.add.at(C, (j, i), 1)
    return H, C

def flat_prior(m):
    P = np.full((m, m), 0.5)
    np.fill_diagonal(P, 0)
    return P

def last_season_prior(H_prev, C_prev):
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(C_prev > 0, H_prev / C_prev, 0.5)
    np.fill_diagonal(P, 0)
    return P

def espn_prior(m):
    s = np.array([espn_projections[t] for t in teams])
    P = s[:, None] / (s[:, None] + s[None, :])
    np.fill_diagonal(P, 0)
    return P

def build_A(H, C, P, w):
    n = C.sum(axis=1)
    A = (H + w * P) / (n + w * (m - 1))[:, None]
    np.fill_diagonal(A, 0)
    return A

def eigen(A):
    ev, vecs = np.linalg.eig(A)
    order = np.argsort(np.abs(ev))[::-1]
    lam = ev[order[0]].real
    r = np.abs(vecs[:, order[0]].real)
    r /= r.sum()
    return lam, np.abs(ev[order[1]]) / lam, r

#run
w = 1
H26, C26 = fill_matricies(g26, m)
H25, C25 = fill_matricies(g25, m)

priors = {
    "flat": flat_prior(m),
    "last season": last_season_prior(H25, C25),
    "espn": espn_prior(m),
}

results = {}
for name, P in priors.items():
    A = build_A(H26, C26, P, w)
    lam, ratio, r = eigen(A)
    results[name] = r
    print(f"{name} prior: lambda* = {lam:.4f}, |lambda2/lambda*| = {ratio:.4f}")

#bar chart
fig, axes = plt.subplots(1, len(results), figsize=(18, 6))
for ax, (name, r) in zip(axes, results.items()):
    order = np.argsort(r)[::-1]
    ax.bar([teams[k] for k in order], r[order])
    ax.set_title(f"{name} prior")
    ax.set_ylabel("r")
    ax.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.savefig("problem_2.png")
plt.show()

#predictions
for name, r in results.items():
    print(f"\n{name} prior predictions:")
    correct = 0
    played = 0
    for _, g in gpred.iterrows():
        pick = g.home if r[g.i] > r[g.j] else g.away
        line = f"  {g.away} @ {g.home}: pick {pick}"
        if g.completed == 1:
            winner = g.home if g.home_pts > g.away_pts else g.away
            played += 1
            correct += pick == winner
            line += f" (actual {winner})"
        print(line)
    if played > 0:
        print(f"  {correct}/{played} correct")