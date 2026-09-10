import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng(5)

# Sampling
def sample_H(N):
    U = rng.random(N)
    Z = 1 - np.sqrt(1 - U)
    return Z

def sample_C(N, R):
    C = np.ones(R)
    for i in tqdm(range(3, N)):
        U = rng.random(R)
        prob = C / i
        C += (U < prob)
    return C

# Parameters
N_h = 10**5
N = 10**4
R = 10**3

# Plotting
z = np.linspace(0, 1, 500)
h = 2 * (1 - z)
Z = sample_C(N, R) / N

fig, axs = plt.subplots()
axs.hist(Z, bins=np.linspace(0.0, 1.0, 40), density=True, label='Empirical pdf')
axs.plot(z, h, label='True pdf h(z)')

axs.set_xlabel("z = C/N")
axs.set_ylabel("probability density")
axs.legend()
fig.savefig("problem_5.png")

print(f'h(z) empirical mean: {sample_H(N_h).mean()}')
print(f'h(z) theoretical mean: {1/3}\n')

print(f'Empirical mean: {Z.mean()}')
print(f'Theoretical mean: {1/3}')
print(f'Empirical sd/mean: {Z.std() / Z.mean()}')
print(f'Theoretical sd/mean: {1/np.sqrt(2)}')
print(f'Smallest core: {round(Z.min() * N)}')
print(f'Largest core: {round(Z.max() * N)}')

plt.show()
    
