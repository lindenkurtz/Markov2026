import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

#part b
Q = np.array([[-2, 2, 0, 0],
             [1, -3, 2, 0],
             [0, 2, -4, 2],
             [0, 0, 3, -3]])

eigenvalues = np.linalg.eigvals(Q)
print(f'eigenvalues: {eigenvalues}')

#part c
x_0 = np.array([1, 0, 0, 0])
pi = np.array([3, 6, 6, 4]) / 19
lam_2 = np.sort(eigenvalues.real)[-2]

def x_t(t):
    return x_0 @ expm(t * Q)

def TV(t):
    return (0.5) * np.sum(np.abs(x_t(t) - pi))

ts = np.linspace(0, 5, 500)
xs = np.array([x_t(t) for t in ts])
tvs = np.array([TV(t) for t in ts])

for i in range(4):
    plt.plot(ts, xs[:, i], label=f'x_{i}(t)')
plt.plot(ts, tvs, color='black', label='TV(t)')
plt.plot(ts, tvs[-1] * np.exp(lam_2 * (ts - 5)), color='red', linestyle='--', label=f'slope lambda_2 = {lam_2:.4f}')
plt.yscale('log')
plt.ylim(1e-4, 1.5)
plt.xlabel('t')
plt.legend()
plt.savefig('problem3c.png')

t_mix = brentq(lambda t: TV(t) - 0.01, 0, 5)
print(f'TV = 0.01 at t = {t_mix:.4f}  ln(100)/|lambda_2| = {np.log(100) / abs(lam_2):.4f}')
