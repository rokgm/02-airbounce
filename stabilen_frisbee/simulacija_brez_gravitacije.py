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

def minima(lst):
    i = np.argmin(lst)
    if i not in [0, 1, 2] and i != len(lst) - 1:
        return i
    return None

m = 0.175
A = 0.0616
ro = 1.23
K = A * ro / (2 * m)
stall_angle = np.pi / 180 * 25
C_90 = 1.1

C_L = C_L_cutoff(0.138, 2.20, stall_angle)
C_D = C_D_cutoff(0.171, 1.47, C_90)

t_ndim = np.linspace(0, 10, 5000)
v_ndim = 2
alpha_list = np.pi / 180 * np.linspace(0.01, 60., 500)
x_coor_min_list = []
y_coor_min_list = []
alpha_coor_min_list = []

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
for alpha in alpha_list:
    vx0_ndim = v_ndim * np.cos(alpha)
    vy0_ndim = -v_ndim * np.sin(alpha)
    sol_ndim = solution_ndim(t_ndim, K, C_L, C_D, stall_angle, (0, 0, vx0_ndim, vy0_ndim))
    # ax.plot(sol_ndim[:, 0] , sol_ndim[:, 1], '.-')
    minimum = minima(sol_ndim[:,1])
    if minimum != None:
        # ax.plot(sol_ndim[minimum,0], sol_ndim[minimum,1], 'x')
        x_coor_min_list.append(sol_ndim[minimum,0])
        y_coor_min_list.append(-sol_ndim[minimum,1])
        alpha_coor_min_list.append(alpha)

ax.plot(np.array(alpha_coor_min_list) * 180 / np.pi, x_coor_min_list, label=r'$\sigma_1$')
ax.plot(np.array(alpha_coor_min_list) * 180 / np.pi, y_coor_min_list, label=r'$-\sigma_2$')
ax.grid(linestyle='--')
ax.set_xlabel(r'$\alpha [^{\circ}]$')
# ax.set_ylabel(r'$\sigma_1$')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend()
# ax.axis('equal')
# ax.set_title(r'Distance to the Bounce Minimum with angle of attack')
fig.tight_layout()

fig.savefig('predstavitev/distance_to_bounce.pdf')
plt.show()