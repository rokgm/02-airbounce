import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit, minimize

# D je koordinatni sistem diska ob t=0, N je koordinatni sistem x, y

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

def frisbee_D(y, t, C_L, C_D, K):
    d1, d2, v1, v2 = y
    v = (v1**2 + v2**2)**0.5
    alpha = np.arctan(-v2 / v1)
    L1 = K * C_L(alpha) * (-v2) * v
    L2 = K * C_L(alpha) * v1 * v
    D1 = K * C_D(alpha) * (-v1) * v
    D2 = K * C_D(alpha) * (-v2) * v
    a1 = L1 + D1
    a2 = L2 + D2
    dydt = [v1, v2, a1, a2]
    return dydt

def solution(t, K, C_L, C_D, initial):
    D_sistem = odeint(frisbee_D, initial, t, args=(C_L, C_D, K))
    return D_sistem


def plot_C_koef(axes, C_L0, C_Lalpha, stall_angle, C_D0, C_Dalpha, C_90):
    x = np.linspace(0, 1.5, 100)
    y = [C_L_cutoff(C_L0, C_Lalpha, stall_angle)(x) for x in x]
    axes.plot(x * 180 / np.pi, y, label='$C_L$')
    z = [C_D_cutoff(C_D0, C_Dalpha, C_90)(x) for x in x]
    axes.plot(x * 180 / np.pi, z, label='$C_D$')
    axes.set_xlabel('angle of attack [$^\circ$]')
    axes.legend()
    axes.grid(linestyle='--')
    axes.set_title('Lift and drag coefficient')

m = 0.175
A = 0.0616
ro = 1.23
K = A * ro / (2 * m)
stall_angle = np.pi / 180 * 25
C_90 = 1.1

C_L = C_L_cutoff(0.188, 2.37, stall_angle)
C_D= C_D_cutoff(0.15, 1.24, C_90)

t = np.linspace(0, 0.5, 100)
alpha = np.pi / 180 * 12
v = 15
vx0 = v * np.cos(alpha)
vy0 = -v * np.sin(alpha)

sol = solution(t, K, C_L, C_D, (0, 0, vx0, vy0))

ax2.plot(N_sistem[:, 0], N_sistem[:, 1], '.', label='simulation')
    ax2.grid(linestyle='--')
    ax2.legend(fancybox=False, prop={'size':9})
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    #ax2.axis('equal')
    ax2.set_title('Trajectory')

plt.show()