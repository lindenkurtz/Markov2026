import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)

rng = np.random.default_rng(5)

#part a
p = np.array([[0, 1/2, 1/2, 0, 0, 0, 0, 0],
              [0, 0, 1/2, 1/2, 0, 0, 0, 0],
              [1/2, 0, 0, 0, 1/2, 0, 0, 0],
              [1/3, 0, 1/3, 0, 1/3, 0, 0, 0],
              [0, 1/3, 0, 0, 0, 1/3, 1/3, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 0]])
q_0 = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])

q_100 = q_0 @ np.linalg.matrix_power(p, 100)
q_101 = q_0 @ np.linalg.matrix_power(p, 101)

print(f'a) q_100: {q_100}')
print(f'a) q_101: {q_101}')

#part b
d = 0.85
G = d * p + ((1-d)/8)*np.ones((8, 8))
q_n = q_0
q_n1 = q_n @ G
i = 1

while (np.sum(np.abs(q_n1 - q_n)) > 1e-10):
    q_n = q_n1
    q_n1 = q_n @ G
    i += 1

pi_input = (G - np.eye(8)).transpose()
pi_input[-1, :] = 1
b = np.zeros(8)
b[-1] = 1

pi = np.linalg.solve(pi_input, b)

print(f'b) brute force stationary distribution: {q_n1}')
print(f'b) linear solve stationary distribution: {pi}')
print(f'b) iterations: {i}')

#part c
def run(R, G, T, pi, checkpoints, rng):
    cumG = np.cumsum(G, axis=1)
    cumG[:, -1] = 1.0
    states = np.zeros(R, dtype=int)
    counts = np.zeros((R, 8))
    rows = np.arange(R)
    is_cp = np.zeros(T + 1, dtype=bool)
    is_cp[checkpoints] = True
    errs = []

    for t in range(1, T + 1):
        U = rng.random(R)
        states = (U[:, None] < cumG[states]).argmax(axis=1)
        counts[rows, states] += 1
        if is_cp[t]:
            err = np.abs(counts / t - pi).max(axis=1)
            errs.append(np.sqrt(np.mean(err**2)))
    return counts / T, np.array(errs)


R, T = 100, 10**5
checkpoints = np.unique(np.logspace(0, 5, 60).astype(int))
pi_hat, errs = run(R, G, T, pi, checkpoints, rng)
print(f'c) pi approximation: {np.mean(pi_hat, axis=0)}')
print(f'c) surfer 0 estimate: {pi_hat[0]}')

x = np.arange(1, 9)
plt.figure()
plt.bar(x - 0.2, pi_hat[0], 0.4, label='surfer 0')
plt.bar(x + 0.2, pi, 0.4, label=r'$\pi$')
plt.xlabel('page'); plt.legend()
plt.savefig('problem_3_a.png', dpi=300)

mask = checkpoints >= 100
slope, intercept = np.polyfit(np.log10(checkpoints[mask]), np.log10(errs[mask]), 1)
plt.figure()
plt.loglog(checkpoints, errs, 'o-', ms=3, label='rms max error')
plt.loglog(checkpoints[mask], 10**intercept * checkpoints[mask]**slope, '--', label=f'slope {slope:.2f}')
plt.xlabel('T'); plt.legend()
plt.savefig('problem_3_b.png', dpi=300)
plt.show()