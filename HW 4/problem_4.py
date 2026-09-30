import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

#functions
def chain(state):
    # returns a touple representing the next state and if the chain has been absorbed
    # 1 if absorbed, 0 otherwise.
    if (state == 'U'):
        Unif = rng.random()
        if (Unif < 0.5):
            return 'I', 0
        else:
            return 'M', 0
    if (state == 'I'):
        Unif = rng.random()
        if (Unif < 0.5):
            return 'F', 1
        elif (Unif < 0.75):
            return 'A', 1
        else:
            return 'U', 0
    if (state == 'M'):
        Unif = rng.random()
        if (Unif < 0.25):
            return 'A', 1
        else:
            return 'U', 0
    if (state == 'F'):
        return 'F', 1
    if (state == 'A'):
        return 'A', 1
    raise ValueError("Invalid State")

vec_chain = np.vectorize(chain)

def run_simulation(starting_state, R = 10 ** 4):
    if starting_state not in {'U', 'I', 'M', 'F', 'A'}:
        raise ValueError("Invalid starting state")
    chains = np.full(R, starting_state, dtype='U1')
    absorbtions = np.zeros(R)
    times = np.zeros(R)

    while (np.any(absorbtions == 0)):
        times = times + (absorbtions == 0)
        chains, absorbtions = vec_chain(chains)

    return chains, times

#simulation
chains_U, times_U = run_simulation('U')
chains_I, times_I = run_simulation('I')
chains_M, times_M = run_simulation('M')

h_U = np.mean(chains_U == 'F')
h_I = np.mean(chains_I == 'F')
h_M = np.mean(chains_M == 'F')

g_U = np.mean(times_U)
g_I = np.mean(times_I)
g_M = np.mean(times_M)

tauF_U = np.mean(times_U[chains_U == 'F'])
tauA_U = np.mean(times_U[chains_U == 'A'])
tauF_I = np.mean(times_I[chains_I == 'F'])
tauA_I = np.mean(times_I[chains_I == 'A'])
tauF_M = np.mean(times_M[chains_M == 'F'])
tauA_M = np.mean(times_M[chains_M == 'A'])

#results
Q = np.array([[0,    0.5, 0.5],
              [0.25, 0,   0  ],
              [0.75, 0,   0  ]])   # rows/cols: U, I, M
R = np.array([[0,   0   ],
              [0.5, 0.25],
              [0,   0.25]])        # cols: F, A

N = np.linalg.inv(np.eye(3) - Q)
h_exact = (N @ R)[:, 0]
g_exact = N @ np.ones(3)
tauF_exact = (N @ N @ R)[:, 0] / h_exact
tauA_exact = (N @ N @ R)[:, 1] / (1 - h_exact)

h_hat = [h_U, h_I, h_M]
g_hat = [g_U, g_I, g_M]
tauF_hat = [tauF_U, tauF_I, tauF_M]
tauA_hat = [tauA_U, tauA_I, tauA_M]

print(f"{'x':<3}{'h':>8}{'h_hat':>8}{'g':>8}{'g_hat':>8}{'tauF':>8}{'tauF_hat':>10}{'tauA':>8}{'tauA_hat':>10}")
for i, x in enumerate(['U', 'I', 'M']):
    print(f"{x:<3}{h_exact[i]:8.4f}{h_hat[i]:8.4f}{g_exact[i]:8.4f}{g_hat[i]:8.4f}"
          f"{tauF_exact[i]:8.4f}{tauF_hat[i]:10.4f}{tauA_exact[i]:8.4f}{tauA_hat[i]:10.4f}")

#plot (start I)
n_max = int(times_I.max())
n = np.arange(1, n_max + 1)
bins = np.arange(0.5, n_max + 1.5)   # width-1 bins centered on integers

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for j, (fate, name, P_fate) in enumerate([('F', 'folded', h_exact[1]), ('A', 'aggregated', 1 - h_exact[1])]):
    pmf = np.array([(np.linalg.matrix_power(Q, k - 1) @ R)[1, j] for k in n]) / P_fate
    axes[j].hist(times_I[chains_I == fate], bins=bins, density=True, alpha=0.5, label='simulated')
    axes[j].plot(n, pmf, 'o', label='exact')
    axes[j].set_title(f'Start I, {name}')
    axes[j].set_xlabel('T')
    axes[j].set_ylabel('P(T = n | fate)')
    axes[j].legend()

plt.tight_layout()
plt.show()
fig.savefig("problem_4.png")

