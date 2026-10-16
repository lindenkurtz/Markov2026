import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

#part c
def h1(R0, N, K):
    k = np.arange(1, K)
    rho = np.cumprod(N / (R0 * (N - k)))
    return 1 / (1 + np.sum(rho))

R0s = np.linspace(0.5, 3, 200)

for N in [100, 1000]:
    K = N // 10
    plt.plot(R0s, [h1(R0, N, K) for R0 in R0s], label=f'exact $h_1$, N = {N}')

plt.plot(R0s, np.maximum(0, 1 - 1/R0s), 'k--', label='$\\max(0, 1 - 1/R_0)$')
plt.xlabel('$R_0$')
plt.ylabel('$h_1$')
plt.title('Probability one case reaches K = N/10')
plt.legend()
plt.savefig('problem_3c.png')
plt.show()

#part d
N, K, gamma, dt = 100, 10, 1, 0.005

def p_q(R0):
    n = np.arange(K + 1)
    beta = R0 * gamma / N
    return beta * n * (N - n) * dt, gamma * n * dt

def expected_T(R0):
    p, q = p_q(R0)
    A = np.zeros((K - 1, K - 1))
    for i in range(K - 1):
        n = i + 1
        A[i, i] = p[n] + q[n]
        if i + 1 < K - 1:
            A[i, i + 1] = -p[n]
        if i - 1 >= 0:
            A[i, i - 1] = -q[n]
    return np.linalg.solve(A, np.ones(K - 1))

for R0 in [0.5, 0.8, 1.5, 3]:
    e = expected_T(R0)
    print(f'd) R0 = {R0}: e_1 * dt = {e[0] * dt:.4f}')

R0 = 1.5
p, q = p_q(R0)
sims = 5000
times = np.zeros(sims)
for s in range(sims):
    x, steps = 1, 0
    while 0 < x < K:
        u = rng.random()
        if u < p[x]:
            x += 1
        elif u < p[x] + q[x]:
            x -= 1
        steps += 1
    times[s] = steps * dt

print(f'd) simulated e_1 * dt for R0 = {R0}: {times.mean():.4f}')
