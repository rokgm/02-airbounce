import matplotlib.pyplot as plt
import numpy as np
from numpy.core.function_base import linspace
from scipy.integrate import odeint

g = 1     # za napalen g se odbije, popravi parametre, video na linku, interpolacija
m = 0.175
A = 0.057
ro = 1.23
C_L0 = 0.15
C_Lalpha = 1.4
C_D0 = 0.08
C_Dalpha = 2.72
K = A * ro / (2 * m)
theta = np.pi / 12
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
R0R0 = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna bločna

# C_L0 = 0.0
# C_Lalpha = 0.0
# C_D0 = 0.0
# C_Dalpha = 0.0

def frisbee_D(y, t, C_L0, C_Lalpha, C_D0, C_Dalpha, K, theta, g):
    d1, d2, v1, v2 = y
    v = (v1**2 + v2**2)**0.5
    alpha = np.arctan(-v2 / v1)
    L1 = K * (C_L0 + C_Lalpha * alpha) * (-v2) * v
    L2 = K * (C_L0 + C_Lalpha * alpha) * v1 * v
    D1 = K * (C_D0 + C_Dalpha * alpha**2) * (-v1) * v
    D2 = K * (C_D0 + C_Dalpha * alpha**2) * (-v2) * v
    a1 = L1 + D1 - g * np.sin(theta)
    a2 = L2 + D2 - g * np.cos(theta)
    dydt = [v1, v2, a1, a2]
    return dydt

def frisbee_N(sol):
    return np.matmul(R0R0, sol.T).T

def initial_N_to_D(d1, d2, v1, v2):
    return np.matmul(R0R0.T, np.array([d1, d2, v1, v2])).T

y0 = initial_N_to_D(0, 0, 15, -15)         # zacetne v D ali N sistemu, v1 v D >= 0 drugace se obrne
t = linspace(0, 10, 101)
sol = odeint(frisbee_D, y0, t, args=(C_L0, C_Lalpha, C_D0, C_Dalpha, K, theta, g))

N_sistem = frisbee_N(sol)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize = (6, 8))
ax1.plot(t, sol[:, 0], label='d1')
ax1.plot(t, sol[:, 1], label='d2')
ax1.plot(t, sol[:, 2], label='v1')
ax1.plot(t, sol[:, 3], label='v2')
ax1.legend(loc='best')
ax1.grid(linestyle='--')
ax1.legend(fancybox=False, edgecolor='black')

ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(d1, d2), N sistem')
ax2.plot(sol[:, 0], sol[:, 1], label='(d1, d2), D sistem')
ax2.legend(loc='best')
ax2.grid(linestyle='--')
ax2.legend(fancybox=False, edgecolor='black')

ax3.plot(N_sistem[:, 2], N_sistem[:, 3], label='(v1, v2), N sistem')
ax3.plot(sol[:, 2], sol[:, 3], label='(v1, v2), D sistem')
ax3.legend(loc='best')
ax3.grid(linestyle='--')
ax3.legend(fancybox=False, edgecolor='black')

plt.tight_layout()
plt.show()