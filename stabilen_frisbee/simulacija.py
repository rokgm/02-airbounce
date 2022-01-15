import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# D je koordinatni sistem diska ob t=0, N je koordinatni sistem x, y

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

def frisbee_D_to_N(sol, R0_0R):
    return np.matmul(R0_0R, sol.T).T

def initial_N_to_D(d1, d2, v1, v2, R0_0R):
    return np.matmul(R0_0R.T, np.array([d1, d2, v1, v2])).T

def plot_1_all(t, K, g, theta, C_dict, inital, inital_in_N=True):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R0_0R = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna blocna

    C_L0, C_Lalpha, C_D0, C_Dalpha = C_dict['C_L0'], C_dict['C_Lalpha'], C_dict['C_D0'], C_dict['C_Dalpha']

    # zacetne v D ali N sistemu, v1 v D >= 0 drugace se obrne
    x, y, vx, vy = inital
    if inital_in_N:
        y0 = initial_N_to_D(x, y, vx, vy, R0_0R)
    else:
        y0 = (x, y, vx, vy)

    sol = odeint(frisbee_D, y0, t, args=(C_L0, C_Lalpha, C_D0, C_Dalpha, K, theta, g))
    N_sistem = frisbee_D_to_N(sol, R0_0R)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize = (6, 8))
    ax1.plot(t, sol[:, 0], label='$d_1$')
    ax1.plot(t, sol[:, 1], label='$d_2$')
    ax1.plot(t, sol[:, 2], label='$v_1$')
    ax1.plot(t, sol[:, 3], label='$v_2$')
    ax1.grid(linestyle='--')
    ax1.legend(fancybox=False, prop={'size':9})

    ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), N sistem')
    ax2.plot(sol[:, 0], sol[:, 1], label='$(d_1, d_2)$, D sistem')
    ax2.grid(linestyle='--')
    ax2.legend(fancybox=False, prop={'size':9})
    ax2.set_xlabel('x, d1 [m]')
    ax2.set_ylabel('y, d2 [m]')
    ax2.axis('equal')


    ax3.plot(N_sistem[:, 2], N_sistem[:, 3], label='$(v_x, v_y)$, N sistem')
    ax3.plot(sol[:, 2], sol[:, 3], label='$(v_{d1}, v_{d2})$, D sistem')
    ax3.grid(linestyle='--')
    ax3.legend(fancybox=False, prop={'size':9})
    ax3.set_xlabel('$v_x, v_{d1}$ [m/s]')
    ax3.set_ylabel('$v_y, v_{d2}$ [m/s]')
    plt.tight_layout()
    plt.show()

def plot_N_trajectories(t, K, g, theta_list, C_list, inital, inital_in_N=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (6, 4))    

    for i in range(len(theta_list)):
        R = np.array([[np.cos(theta_list[i]), -np.sin(theta_list[i])], [np.sin(theta_list[i]), np.cos(theta_list[i])]])
        R0_0R = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])   # diagonalna blocna
        C_L0, C_Lalpha, C_D0, C_Dalpha = C_list[i]['C_L0'], C_list[i]['C_Lalpha'], C_list[i]['C_D0'], C_list[i]['C_Dalpha']
        
        # zacetne v D ali N sistemu, v1 v D >= 0 drugace se obrne
        x, y, vx, vy = inital
        if inital_in_N:
            y0 = initial_N_to_D(x, y, vx, vy, R0_0R)
        else:
            y0 = (x, y, vx, vy)

        sol = odeint(frisbee_D, y0, t, args=(C_L0, C_Lalpha, C_D0, C_Dalpha, K, theta, g))
        N_sistem = frisbee_D_to_N(sol, R0_0R)

        ax.plot(N_sistem[:, 0], N_sistem[:, 1], label='C{}'. format(i + 1))
        ax.legend(fancybox=False, prop={'size':9})
        ax.grid(linestyle='--')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        ax.set_title('N sistem')

    plt.tight_layout()
    plt.show()

g = 9.8

m = 0.175 # competition disc
# m = 0.027   # gravity disc

# A = 0.0616 # competition disc
A = 0.057 # iz clanka
# A = 0.00312 # gravity disc

ro = 1.23
K = A * ro / (2 * m)

# probaj se za C-je, video https://www.youtube.com/watch?v=BTiLOtF-LGY, 5:36, interpolacija
C0_dict = {'C_L0':0.0, 'C_Lalpha':0.0, 'C_D0':0.0, 'C_Dalpha':0.0}
C1_dict = {'C_L0':0.15, 'C_Lalpha':1.4, 'C_D0':0.08, 'C_Dalpha':2.72}   # clanek za_koeficiente1, za_koeficeinte2
C2_dict = {'C_L0':0.188, 'C_Lalpha':2.37, 'C_D0':0.15, 'C_Dalpha':1.24} # clanek identification_of...
C3_dict = {'C_L0':0.2, 'C_Lalpha':2.96, 'C_D0':0.08, 'C_Dalpha':2.60}   # -||-
C4_dict = {'C_L0':-0.40, 'C_Lalpha':1.89, 'C_D0':0.83, 'C_Dalpha':0.83} # clanek frisbee_sim
C5_dict = {'C_L0':1.17, 'C_Lalpha':0.28, 'C_D0':5.07, 'C_Dalpha':0.077} # -||-

t = np.linspace(0, 2, 101)
theta = np.pi / 180 * 20

plot_1_all(t, K, g, theta, C2_dict, (0, 0, 15, -8))

theta_list = np.ones(5) * theta
C_list = [C1_dict, C2_dict, C3_dict, C4_dict, C5_dict]
plot_N_trajectories(t, K, g, theta_list, C_list, (0, 0, 15, -8))