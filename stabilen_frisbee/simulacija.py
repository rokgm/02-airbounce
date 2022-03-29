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

def functional(x, t_eks1, x_eks1, y_eks1, K, g, initial_eks, C90):     # dodaj se stall angle ce dela
    C_L0, C_Lalpha, C_D0, C_Dalpha, stall_angle = x
    C_L = C_L_cutoff(C_L0, C_Lalpha, stall_angle)
    C_D = C_D_cutoff(C_D0, C_Dalpha, C_90)
    N_sistem = solution(t_eks1, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
    distance_sqr = (N_sistem[:, 0] - x_eks1)**2 + (N_sistem[:, 1] - y_eks1)**2
    return np.average(distance_sqr)

g = 9.8
m = 0.175
A = 0.0616
ro = 1.23
K = A * ro / (2 * m)

# EKSPERIMENT #
t_eks1, x_eks1, y_eks1, vx_eks1, vy_eks1 = np.loadtxt('video_analiza_1.dat', unpack=True, max_rows=49)
t_eks1 -= t_eks1[0]
initial_eks = x_eks1[0], y_eks1[0], np.average(vx_eks1[0:3]), np.average(vy_eks1[0:3])
###############

theta = np.pi / 180 * 12
stall_angle = np.pi / 180 * 15
C_90 = 1.1

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize = (6, 7))

# clanek
C_L = C_L_cutoff(0.188, 2.37, stall_angle)  # clanek
C_D = C_D_cutoff(0.15, 1.24, C_90)
plot_C_koef(0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
N_sistem = solution(t_eks1, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
ax1.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
ax1.plot(x_eks1, y_eks1, label='(x, y), eksperiment')
ax1.grid(linestyle='--')
ax1.legend(fancybox=False, prop={'size':8})
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.axis('equal')
ax1.set_title('Članek: C_L0, C_Lalpha, C_D0, C_Dalpha, stall_angle \n [0.188, 2.37, 0.15, 1.24, 0.26]')

# minimizacija: stall, 1 ali 2, 15 ali 25
# method='Nelder-Mead'
C_fit = minimize(functional, (0.188, 2.37, 0.15, 1.24, 0.26), \
    args=(t_eks1, x_eks1, y_eks1, K, g, initial_eks, C_90), method='Nelder-Mead', \
         bounds = ((0.10, 0.22), (1.9, 2.7), (0.10, 0.19), (1.00, 1.40), (0.1, 0.5)), tol=1e-2)
print(C_fit.x)
C_L = C_L_cutoff(C_fit.x[0], C_fit.x[1], stall_angle)  # 
C_D = C_D_cutoff(C_fit.x[2], C_fit.x[3], C_90)
# plot_C_koef(0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
N_sistem = solution(t_eks1, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
ax2.plot(x_eks1, y_eks1, label='(x, y), eksperiment')
ax2.grid(linestyle='--')
ax2.legend(fancybox=False, prop={'size':8})
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
ax2.axis('equal')
ax2.set_title('Minimization method=Nelder-Mead \n {}'.format(C_fit.x))

# method='BFGS'
C_fit = minimize(functional, (0.188, 2.37, 0.15, 1.24, 0.26), \
     args=(t_eks1, x_eks1, y_eks1, K, g, initial_eks, C_90), \
     bounds = ((0.10, 0.22), (1.9, 2.7), (0.10, 0.19), (1.00, 1.40), (0.1, 0.5)), tol=1e-2)
print(C_fit.x)
C_L = C_L_cutoff(C_fit.x[0], C_fit.x[1], stall_angle)  # 
C_D = C_D_cutoff(C_fit.x[2], C_fit.x[3], C_90)
# plot_C_koef(0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
N_sistem = solution(t_eks1, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
ax3.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
ax3.plot(x_eks1, y_eks1, label='(x, y), eksperiment')
ax3.grid(linestyle='--')
ax3.legend(fancybox=False, prop={'size':8})
ax3.set_xlabel('x [m]')
ax3.set_ylabel('y [m]')
ax3.axis('equal')
ax3.set_title('Minimization method=default, BFGS \n {}'.format(C_fit.x))

plt.tight_layout()
plt.show()

# fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (8, 6))
# ax1.plot(t_eks1, N_sistem[:, 0], label='$x$')
# ax1.plot(t_eks1, N_sistem[:, 1], label='$y$')
# ax1.plot(t_eks1, N_sistem[:, 2], label='$v_x$')
# ax1.plot(t_eks1, N_sistem[:, 3], label='$v_y$')
# ax1.grid(linestyle='--')
# ax1.legend(fancybox=False, prop={'size':9})
# ax1.set_title('Simulacija')
# ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
# # EKSPERIMENT #
# ax2.plot(x_eks1, y_eks1, label='(x, y), eksperiment')
# ax2.grid(linestyle='--')
# ax2.legend(fancybox=False, prop={'size':9})
# ax2.set_xlabel('x [m]')
# ax2.set_ylabel('y [m]')
# ax2.axis('equal')
# ax2.set_title('Trajektorija')
# ax3.plot(t_eks1, x_eks1, label='$x$')
# ax3.plot(t_eks1, y_eks1, label='$y$')
# ax3.plot(t_eks1, vx_eks1, label='$v_x$')
# ax3.plot(t_eks1, vy_eks1, label='$v_y$')
# ax3.grid(linestyle='--')
# ax3.legend(fancybox=False, prop={'size':9})
# ax3.set_title('Eksperiment')
# ###############
# plt.tight_layout()
# plt.show()