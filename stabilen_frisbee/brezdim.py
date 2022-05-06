import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def C_L_cutoff(C_L0, C_Lalpha, stall_angle):
    '''Lift cutoff pri stall angle: rad'''
    a = (C_L0 + C_Lalpha * stall_angle) / (1 * (stall_angle - np.pi / 2)**2)

    def C_L(alpha):
        if alpha >= stall_angle:
            return a * (alpha - np.pi / 2)**2
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

def frisbee_D_dim(y, t, C_L, C_D, theta):
    d1, d2, v1, v2 = y
    v = (v1**2 + v2**2)**0.5
    alpha = np.arctan(-v2 / v1)
    L1 = C_L(alpha) * (-v2) * v
    L2 = C_L(alpha) * v1 * v
    D1 = C_D(alpha) * (-v1) * v
    D2 = C_D(alpha) * (-v2) * v
    a1 = L1 + D1 - np.sin(theta)
    a2 = L2 + D2 - np.cos(theta)
    dydt = [v1, v2, a1, a2]
    return dydt

def frisbee_D_to_N(sol, R0_0R):
    return np.matmul(R0_0R, sol.T).T

def initial_N_to_D(d1, d2, v1, v2, R0_0R):
    return np.matmul(R0_0R.T, np.array([d1, d2, v1, v2])).T

def solution_dim(t, K, g, theta, C_L, C_D, stall_angle, inital, inital_in_N=True):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R0_0R = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna blocna
    
    x, y, vx, vy = inital
    if inital_in_N:
        y0 = initial_N_to_D(x, y, vx, vy, R0_0R)
    else:
        y0 = (x, y, vx, vy)

    D_sistem = odeint(frisbee_D_dim, y0, t, args=(C_L, C_D, theta))
    N_sistem = frisbee_D_to_N(D_sistem, R0_0R)
    return N_sistem, D_sistem

def minima(lst):
    i = np.argmin(lst)
    if i not in [0, 1, 2] and i != len(lst) - 1:
        return i
    return None

def curvature(x, y):
    '''1. odvod = 0 v min, zato samo 2. odvod'''
    dy = np.gradient(y, x)
    ddy = np.gradient(dy, x)
    return ddy

g = 9.8
m = 0.175
A = 0.0616
ro = 1.23
K = A * ro / (2 * m)
stall_angle = np.pi / 180 * 25
C_90 = 1.1

D_c = 1 / K
T_c = (1 / (K * g))**0.5

C_L = C_L_cutoff(0.138, 2.20, stall_angle)
C_D = C_D_cutoff(0.171, 1.47, C_90)

theta = np.pi / 180 * 10
t = np.linspace(0, 1, 100)
v_list = np.linspace(0.1, 4, 3)
alpha_list = np.linspace(1, 20, 3) * np.pi / 180

fig, (ax1, ax2) = plt.subplots(2, 1)

minimum_alpha = []
minimum_v = []
minimum_curv = []

label = 0
for v in v_list:
    for alpha in alpha_list:
        vx0 = np.cos(alpha) * v
        vy0 = -np.sin(alpha) * v
        sol = solution_dim(t, K, g, theta, C_L, C_D, stall_angle, (0, 0, vx0, vy0))[0]
        i = minima(sol[:,1])
        if i != None:
            minimum_alpha.append(alpha * 180 / np.pi)
            minimum_v.append(v)
            minimum_curv.append(curvature(sol[i-2:i+3, 0], sol[i-2:i+3, 1])[3])
            ax1.plot(sol[:, 0], sol[:, 1], label='{}'.format(label))
            label += 1

print(minimum_curv)
ax2.scatter(minimum_alpha, minimum_v)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.set_xlabel(r'$\alpha [^{\circ}]$')
ax2.set_ylabel(r'$\nu$')

ax1.set_xlabel(r'$\sigma_x$')
ax1.set_ylabel(r'$\sigma_y$')
ax1.legend()
# ax1.axis('equal')
ax1.set_title('Trajectory')

fig.tight_layout()
plt.show()