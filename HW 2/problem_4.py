import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng(5)

def sampling(N, a, lam_f, lam_s):
    samples = np.zeros(N)
    for i in tqdm(range(N)):
        U1 = rng.random()
        U2 = rng.random()
        if (U1 < a):
            samples[i] = -np.log(1 - U2) / lam_f
        else:
            samples[i] = -np.log(1 - U2) / lam_s
    return samples

# Parameters
N = 10**5
a = 0.9
lam_f = 1000
lam_s = 10

# Plotting
t = np.linspace(0, 0.5, 500)
f = a * lam_f * np.exp(-lam_f * t) + (1-a) * lam_s * np.exp(-lam_s * t)
T = sampling(N, a, lam_f, lam_s)
 
fig, axs = plt.subplots()
axs.hist(T, bins=np.linspace(0.0, 0.5, 150), density=True, label='Empirical pdf')
axs.plot(t, f, label='True pdf f(x)')
 
axs.set_yscale("log")
axs.set_ylim(bottom=1e-3)
axs.set_xlabel("time (seconds)")
axs.set_ylabel("probability density")
axs.legend()
fig.savefig("problem_4.png")

print(f'Empirical mean: {T.mean()} seconds')
print(f'Theoretical mean: 0.0109 seconds')
print(f'Empirical P(T > 50 ms): {(T > .05).mean()}')
print(f'Theoretical P(T > 50 ms): 0.0607')

plt.show()