import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from warnings import warn

# D je koordinatni sistem diska ob t=0, N je koordinatni sistem x, y

def frisbee_D(y, t, C_L, C_D, K, theta, g, stall_angle):
    d1, d2, v1, v2 = y
    v = (v1**2 + v2**2)**0.5
    alpha = np.arctan(-v2 / v1)

    if alpha > stall_angle:
        warn('Angle of attack > stall_angle => {}°'.format(stall_angle * 180 / np.pi))

    L1 = K * C_L(alpha) * (-v2) * v
    L2 = K * C_L(alpha) * v1 * v
    D1 = K * C_D(alpha) * (-v1) * v
    D2 = K * C_D(alpha) * (-v2) * v
    a1 = L1 + D1 - g * np.sin(theta)
    a2 = L2 + D2 - g * np.cos(theta)
    dydt = [v1, v2, a1, a2]
    return dydt

def frisbee_D_to_N(sol, R0_0R):
    return np.matmul(R0_0R, sol.T).T

def initial_N_to_D(d1, d2, v1, v2, R0_0R):
    return np.matmul(R0_0R.T, np.array([d1, d2, v1, v2])).T

def plot_1_all(t, K, g, theta, C_L, C_D, stall_angle, inital, inital_in_N=True):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R0_0R = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna blocna

    # C_L0, C_Lalpha, C_D0, C_Dalpha = C_dict['C_L0'], C_dict['C_Lalpha'], C_dict['C_D0'], C_dict['C_Dalpha']

    # zacetne v D ali N sistemu, v1 v D >= 0 drugace se obrne
    x, y, vx, vy = inital
    if inital_in_N:
        y0 = initial_N_to_D(x, y, vx, vy, R0_0R)
    else:
        y0 = (x, y, vx, vy)

    sol = odeint(frisbee_D, y0, t, args=(C_L, C_D, K, theta, g, stall_angle))
    N_sistem = frisbee_D_to_N(sol, R0_0R)

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (8, 6))
    ax1.plot(t, N_sistem[:, 0], label='$x$')
    ax1.plot(t, N_sistem[:, 1], label='$y$')
    ax1.plot(t, N_sistem[:, 2], label='$v_x$')
    ax1.plot(t, N_sistem[:, 3], label='$v_y$')
    ax1.grid(linestyle='--')
    ax1.legend(fancybox=False, prop={'size':9})
    ax1.set_title('Simulacija')

    ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
    # ax2.plot(sol[:, 0], sol[:, 1], label='$(d_1, d_2)$, D sistem')

    # ax3.plot(N_sistem[:, 2], N_sistem[:, 3], label='$(v_x, v_y)$, N sistem')
    # ax3.plot(sol[:, 2], sol[:, 3], label='$(v_{d1}, v_{d2})$, D sistem')
    # ax3.grid(linestyle='--')
    # ax3.legend(fancybox=False, prop={'size':9})
    # ax3.set_xlabel('$v_x, v_{d1}$ [m/s]')
    # ax3.set_ylabel('$v_y, v_{d2}$ [m/s]')

    # EKSPERIMENT #
    ax2.plot(x_eks1, y_eks1, label='(x, y), eksperiment')
    ax2.grid(linestyle='--')
    ax2.legend(fancybox=False, prop={'size':9})
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.axis('equal')
    ax2.set_title('Trajektorija')

    ax3.plot(t_eks1, x_eks1, label='$x$')
    ax3.plot(t_eks1, y_eks1, label='$y$')
    ax3.plot(t_eks1, vx_eks1, label='$v_x$')
    ax3.plot(t_eks1, vy_eks1, label='$v_y$')
    ax3.grid(linestyle='--')
    ax3.legend(fancybox=False, prop={'size':9})
    ax3.set_title('Eksperiment')
    ###############
    plt.tight_layout()


def plot_N_trajectories(t, K, g, theta_list, C_L_list, C_D_list, stall_angle_list, inital_list, inital_in_N=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (6, 4))  

    for i in range(len(theta_list)):
        R = np.array([[np.cos(theta_list[i]), -np.sin(theta_list[i])], [np.sin(theta_list[i]), np.cos(theta_list[i])]])
        R0_0R = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna blocna
        # C_L0, C_Lalpha, C_D0, C_Dalpha = C_list[i]['C_L0'], C_list[i]['C_Lalpha'], C_list[i]['C_D0'], C_list[i]['C_Dalpha']
        
        # zacetne v D ali N sistemu, v1 v D >= 0 drugace se obrne
        x, y, vx, vy = inital_list[i]
        if inital_in_N:
            y0 = initial_N_to_D(x, y, vx, vy, R0_0R)
        else:
            y0 = (x, y, vx, vy)

        sol = odeint(frisbee_D, y0, t, args=(C_L_list[i], C_D_list[i], K, theta_list[i], g, stall_angle_list[i]))
        N_sistem = frisbee_D_to_N(sol, R0_0R)
        ax.plot(N_sistem[:, 0], N_sistem[:, 1], label=r'$\theta$ = {}'. format((i) * 5))   # prilagodi

    ax.legend(fancybox=False, prop={'size':9})
    ax.grid(linestyle='--')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis('equal')
    ax.set_title('N sistem')     
    plt.tight_layout()
    # plt.savefig('med_razlicni_koti.png', dpi=600, bbox_inches='tight')

def C_L_cutoff(C_L0, C_Lalpha, stall_angle):
    '''Lift cutoff pri stall angle: rad'''
    a = (C_L0 + C_Lalpha * stall_angle) / (1 * (stall_angle - np.pi / 2)**2)

    def C_L(alpha):
        if alpha >= stall_angle:
            return a * (alpha - np.pi / 2)**2 ###### 2
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

def plot_C_koef(C_L0, C_Lalpha, stall_angle, C_D0, C_Dalpha, C_90):
    x = np.linspace(0, 1.5, 100)
    y = [C_L_cutoff(C_L0, C_Lalpha, stall_angle)(x) for x in x]
    plt.plot(x * 180 / np.pi, y, label='$C_L$')
    z = [C_D_cutoff(C_D0, C_Dalpha, C_90)(x) for x in x]
    plt.plot(x * 180 / np.pi, z, label='$C_D$')
    plt.xlabel('angle of attack [$^\circ$]')
    plt.legend()
    plt.title('C koeficienta, manj oster stall cutoff')
    # plt.savefig('koeficienta_cutoff.png', dpi=600, bbox_inches='tight')

g = 9.8
m = 0.175 # competition disc
# m = 0.027   # gravity disc
A = 0.0616 # competition disc
# A = 0.057 # iz clanka
# A = 0.00312 # gravity disc

ro = 1.23
K = A * ro / (2 * m)

# https://www.youtube.com/watch?v=BTiLOtF-LGY, 5:36, interpolacija
# alpha = np.array([0, 10, 20, 30, 40, 45]) * np.pi / 180
# C_L = np.array([5.5, 8.56, 13.5, 17.6, 21.2, 28.75])
# C_D = np.array([0.1, 0.25, 0.35, 0.41, 0.44, 0.68])

C0_dict = {'C_L0':0.0, 'C_Lalpha':0.0, 'C_D0':0.0, 'C_Dalpha':0.0}
C1_dict = {'C_L0':0.15, 'C_Lalpha':1.4, 'C_D0':0.08, 'C_Dalpha':2.72}   # (slaba napoved), clanek za_koeficiente1, za_koeficeinte2
C2_dict = {'C_L0':0.188, 'C_Lalpha':2.37, 'C_D0':0.15, 'C_Dalpha':1.24} # clanek identification_of...
C3_dict = {'C_L0':0.2, 'C_Lalpha':2.96, 'C_D0':0.08, 'C_Dalpha':2.60}   # (slaba napoved), -||-
C4_dict = {'C_L0':-0.40, 'C_Lalpha':1.89, 'C_D0':0.83, 'C_Dalpha':0.83} # (slaba napoved), clanek frisbee_sim
C5_dict = {'C_L0':1.17, 'C_Lalpha':0.28, 'C_D0':5.07, 'C_Dalpha':0.077} # (slaba napoved), -||-

# EKSPERIMENT #
t_eks1, x_eks1, y_eks1, vx_eks1, vy_eks1 = np.loadtxt('video_analiza_1.dat', unpack=True, max_rows=49)
t_eks1 -= t_eks1[0]
###############

t = np.linspace(0, 1, 101)

theta = np.pi / 180 * 12
stall_angle = np.pi / 180 * 25
C_90 = 1.1
C_L = C_L_cutoff(0.188, 2.37, stall_angle)
C_D = C_D_cutoff(0.15, 1.24, C_90)
plot_C_koef(0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
plot_1_all(t, K, g, theta, C_L, C_D, stall_angle, (0, 0, 14, -3.8))
plt.show()

# theta_list = np.ones(7) * np.pi * 25 / 180
# theta_list = np.linspace(0, 30, 7) * np.pi / 180
# # stall_angle_list = np.pi / 180 * np.linspace(10, 40, 5)     # nastavljeno 1
# stall_angle_list = np.ones(7) * np.pi / 180 * 25
# C_90_list = np.ones(7) * 1.1
# C_L_list = [(0.2, 2.96) for _ in theta_list]
# C_D_list = [(0.08, 2.60) for _ in theta_list]
# initial_list = [(0, 0, 15, -8) for _ in theta_list]
# # plot_C_koef(0.2, 2.96, stall_angle, 0.08, 2.60, C_90)
# C_L_list = [(C_L_list[i][0], C_L_list[i][1], stall_angle_list[i]) for i in range(len(theta_list))]
# C_D_list = [(C_D_list[i][0], C_D_list[i][1], stall_angle_list[i]) for i in range(len(theta_list))]
# C_L_list = [C_L_cutoff(*x) for x in C_L_list]
# C_D_list = [C_D_cutoff(*x) for x in C_D_list]
# plot_N_trajectories(t, K, g, theta_list, C_L_list, C_D_list, stall_angle_list, initial_list)