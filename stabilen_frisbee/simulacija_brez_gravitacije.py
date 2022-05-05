import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit, minimize

def C_L_cutoff(C_L0, C_Lalpha, stall_angle):
    '''Lift cutoff pri stall angle: rad'''
    a = (C_L0 + C_Lalpha * stall_angle) / (1 * (stall_angle - np.pi / 2)**2)

    def C_L(alpha):
        if alpha >= stall_angle:
            return a * (alpha - np.pi / 2)**2 / 1 ###### 1
        else:
            return C_L0 + C_Lalpha * alpha
    return C_L

def C_D_cutoff(C_D0, C_Dalpha, C_90):
    '''Drag cutoff pri alpha, ko C_D = C_90, C_90 = za disk pri kotu 90°'''
    alpha_cutoff = ((C_90 - C_D0) / C_Dalpha)**0.5

    def C_D(alpha):
        if alpha >= alpha_cutoff:
            return C_90
        else:
            return C_D0 + C_Dalpha * alpha**2            
    return C_D

def frisbee_D_ndim(y, t, C_L, C_D):
    d1, d2, v1, v2 = y
    v = (v1**2 + v2**2)**0.5
    alpha = np.arctan(-v2 / v1)
    L1 = C_L(alpha) * (-v2) * v
    L2 = C_L(alpha) * v1 * v
    D1 = C_D(alpha) * (-v1) * v
    D2 = C_D(alpha) * (-v2) * v
    a1 = L1 + D1
    a2 = L2 + D2
    dydt = [v1, v2, a1, a2]
    return dydt

def solution_ndim(t, K, C_L, C_D, stall_angle, inital, inital_in_N=True):
    y0 = inital
    D_sistem = odeint(frisbee_D_ndim, y0, t, args=(C_L, C_D))
    return D_sistem

m = 0.175
A = 0.0616
ro = 1.23
K = A * ro / (2 * m)
stall_angle = np.pi / 180 * 25
C_90 = 1.1
C_L = C_L_cutoff(0.138, 2.20, stall_angle)
C_D = C_D_cutoff(0.171, 1.47, C_90)

D_c = 1 / K
T_c = 1

t = np.linspace(0, 10, 100)
t_ndim = t / T_c
v_ndim_list = [4]
alpha_list = np.pi / 180 * np.linspace(-3.5, 50, 10)

kot_nevt = np.pi /180 * 3.5
R = np.array([[np.cos(kot_nevt), -np.sin(kot_nevt)], [np.sin(kot_nevt), np.cos(kot_nevt)]])

fig, ax = plt.subplots()
for alpha in alpha_list:
    for v_ndim in v_ndim_list:
        vx0_ndim = v_ndim * np.cos(alpha)
        vy0_ndim = -v_ndim * np.sin(alpha)
        sol_ndim = solution_ndim(t_ndim, K, C_L, C_D, stall_angle, (0, 0, vx0_ndim, vy0_ndim))
        sol_ndim = np.matmul(R.T, np.array([sol_ndim[:, 0], sol_ndim[:, 1]])).T
        ax.plot(sol_ndim[:, 0] * D_c, sol_ndim[:, 1] * D_c, '.')

ax.grid(linestyle='--')
ax.set_xlabel(r'$\sigma_1$')
ax.set_ylabel(r'$\sigma_2$')
# ax.axis('equal')
ax.set_title('Trajectory')
fig.tight_layout()

plt.show()