import numpy as np

M_arr = []
m_arr = []
Eigen_arr = []

for m in range(3, 13):
    p = np.eye(m)
    p = np.roll(p, shift=1, axis=1)
    p[m-1, 0], p[m-1, 1] = 1/2, 1/2
    i = 1
    b = p > 0
    while not b.all():
        b = b @ p
        b = b > 0
        i += 1
    M_arr.append(i)
    m_arr.append((m-1) ** 2 + 1)
    ev = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
    Eigen_arr.append(round(float(ev[1]), 4))

print("M_arr:", M_arr)
print("m_arr:", m_arr)
print("Eigen_arr:", Eigen_arr)