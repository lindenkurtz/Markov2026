import numpy as np
import matplotlib.pyplot as plt

#part b
p = np.array([[0, 1, 0, 0, 0], 
              [1/4, 0, 3/4, 0, 0], 
              [0, 1/2, 0, 1/2, 0], 
              [0, 0, 3/4, 0, 1/4], 
              [0, 0, 0, 1, 0]])
q_0 = np.array([1, 0, 0, 0, 0])

q_50 = q_0 @ np.linalg.matrix_power(p, 50)
q_51 = q_0 @ np.linalg.matrix_power(p, 51)

print(f'b) q_50: {q_50}')
print(f'b) q_51: {q_51}')

Q = np.zeros((61, 5))
Q[0] = q_0
for i in range(1, 61):
    Q[i] = Q[i - 1] @ p


#part c
p2 = np.array([[0.7, 0.3, 0, 0, 0], 
               [.025, .75, .225, 0, 0], 
               [0, .05, .8, .15, 0], 
               [0, 0, .075, .85, .075], 
               [0, 0, 0, .1, .9]])
p2_input = (p2 - np.eye(5)).transpose()
p2_input[-1, :] = 1
b = np.array([0, 0, 0, 0, 1])
pi2 = np.linalg.solve(p2_input, b)

print(f'c) pi2: {pi2}')

Q2 = np.zeros((400, 5))
Q2[0] = q_0
for i in range(1, 400):
    Q2[i] = Q2[i - 1] @ p2
    
err = np.max(np.abs(Q2 - pi2), axis=1)
n_star = np.argmax(err < 1e-6)

eig = np.linalg.eigvals(p2)
moduli = np.sort(np.abs(eig))[::-1]

print(f'c) smallest n: {n_star}')
print(f'c) eigenvalues: {moduli}')

#plot
n = np.arange(Q.shape[0])
running_avg = np.cumsum(Q[:, 2]) / (np.arange(len(Q)) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(n, Q[:, 2], "o-", ms=3, label=r"$q_n(2)$")
ax1.plot(n, Q[:, 4], "s-", ms=3, label=r"$q_n(4)$")
ax1.plot(n, running_avg, "k--", label="running avg")
ax1.set(xlabel="n", ylabel="probability", title="a = b = 1")
n2 = np.arange(len(Q2))
for k, c in [(2, "C0"), (4, "C1")]:
    ax2.plot(n2, Q2[:, k], c, label=rf"$q_n({k})$")
    ax2.axhline(pi2[k], color=c, ls=":", label=rf"$\pi({k})$")
ax2.set(xlabel="n", title="a = 0.3, b = 0.1")
for ax in (ax1, ax2): ax.legend(); ax.grid(alpha=0.3)
plt.show()
fig.savefig("problem_1.png", dpi=150, bbox_inches="tight")