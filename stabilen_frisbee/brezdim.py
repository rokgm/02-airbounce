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

def frisbee_D(y, t, C_L, C_D, K, theta, g, stall_angle):
    d1, d2, v1, v2 = y
    v = (v1**2 + v2**2)**0.5
    alpha = np.arctan(-v2 / v1)
    L1 = K * C_L(alpha) * (-v2) * v
    L2 = K * C_L(alpha) * v1 * v
    D1 = K * C_D(alpha) * (-v1) * v
    D2 = K * C_D(alpha) * (-v2) * v
    a1 = L1 + D1 - g * np.sin(theta)
    a2 = L2 + D2 - g * np.cos(theta)
    dydt = [v1, v2, a1, a2]
    return dydt

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

def solution(t, K, g, theta, C_L, C_D, stall_angle, inital, inital_in_N=True):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R0_0R = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna blocna

    x, y, vx, vy = inital
    if inital_in_N:
        y0 = initial_N_to_D(x, y, vx, vy, R0_0R)
    else:
        y0 = (x, y, vx, vy)

    D_sistem = odeint(frisbee_D, y0, t, args=(C_L, C_D, K, theta, g, stall_angle))
    N_sistem = frisbee_D_to_N(D_sistem, R0_0R)
    return N_sistem, D_sistem

def solution_dim(t, K, g, theta, C_L, C_D, stall_angle, inital, inital_in_N=True):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R0_0R = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna blocna
    
    x, y, vx, vy = inital
    if inital_in_N:
        y0 = initial_N_to_D(x, y, vx, vy, R0_0R)
    else:
        y0 = (x, y, vx, vy)

    D_sistem = odeint(frisbee_D_dim, y0, t, args=(C_L, C_D, theta))
    D_sistem[:, 0:2] = D_c * D_sistem[:, 0:2]
    D_sistem[:, 2:4] = D_c * D_sistem[:, 2:4]
    N_sistem = frisbee_D_to_N(D_sistem, R0_0R)
    return N_sistem, D_sistem

g = 9.8
m = 0.175
A = 0.0616
ro = 1.23
K = A * ro / (2 * m)
stall_angle = np.pi / 180 * 25
C_90 = 1.1

D_c = 1 / K
T_c = (1 / (K * g))**0.5
print(D_c, T_c)

C_L = C_L_cutoff(0.188, 2.37, stall_angle)
C_D= C_D_cutoff(0.15, 1.24, C_90)

t = np.linspace(0, 2, 100)
vx0 = 15
vy0 = -5
theta = np.pi / 180 * 15

t_dim = np.linspace(0, 2, 100) / T_c
vx0_dim = 15 * T_c / D_c
vy0_dim = -5 * T_c / D_c

sol = solution(t, K, g, theta, C_L, C_D, stall_angle, (0, 0, vx0, vy0))[0]
sol_dim = solution_dim(t_dim, K, g, theta, C_L, C_D, stall_angle, (0, 0, vx0_dim, vy0_dim))[0]

fig, ax = plt.subplots()
ax.plot(sol[:, 0], sol[:, 1], '+', label='obi')
ax.plot(sol_dim[:, 0], sol_dim[:, 1], 'x', label='dim')
ax.grid(linestyle='--')
ax.legend(fancybox=False, prop={'size':9})
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
#ax.axis('equal')
ax.set_title('Trajectory')

plt.show()