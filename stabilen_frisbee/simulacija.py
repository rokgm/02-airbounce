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


def plot_C_koef(axes, C_L0, C_Lalpha, stall_angle, C_D0, C_Dalpha, C_90):
    x = np.linspace(0, 1.5, 100)
    y = [C_L_cutoff(C_L0, C_Lalpha, stall_angle)(x) for x in x]
    axes.plot(x * 180 / np.pi, y, label='$C_L$')
    z = [C_D_cutoff(C_D0, C_Dalpha, C_90)(x) for x in x]
    axes.plot(x * 180 / np.pi, z, label='$C_D$')
    axes.set_xlabel('angle of attack [$^\circ$]')
    axes.legend()
    axes.grid(linestyle='--')
    axes.set_title('C koeficienta')

def funkcional(x, theta, t_eks1, x_eks1, y_eks1, K, g, initial_eks, C_90, stall_angle, weighted):
    C_L0, C_Lalpha, C_D0, C_Dalpha= x
    C_L = C_L_cutoff(C_L0, C_Lalpha, stall_angle)
    C_D = C_D_cutoff(C_D0, C_Dalpha, C_90)
    N_sistem = solution(t_eks1, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
    
    if weighted:
        weights = np.exp(-t_eks1 * 3.)
    else:
        weights = 1.

    distance = weights * ((N_sistem[:, 0] - x_eks1)**2 + (N_sistem[:, 1] - y_eks1)**2)
    return np.average(distance)

def paralaksa_lin(k, x):
    '''
    dx' je pravi, dx izmerjeni
    k = koncna velikost frizbija / zacetna
    l = dolzina meta
    dx = dx' * [ (k - 1) / l * x + 1 ]
    x = \integral dx' = \integral dx / [ (k - 1) / l * x + 1 ]
    '''
    l = x[-1]
    return l / (k - 1) * (np.log((k - 1) * x + l) - np.log(l))

g = 9.8
m = 0.175
A = 0.0616
ro = 1.23
K = A * ro / (2 * m)
stall_angle = np.pi / 180 * 25
C_90 = 1.1

## Podatki ##     paralksa ni vpostevana pri hitrostih, za zacetne pogoje nima veliko vpliva
# video_1
t_eks_1, x_eks_1, y_eks_1, vx_eks_1, vy_eks_1 = np.loadtxt('video_analiza_1.dat', unpack=True, max_rows=49)  # 49 max
t_eks_1 -= t_eks_1[0]
x_eks_1 -= x_eks_1[0]
y_eks_1 -= y_eks_1[0]
faktor_paralakse_1 = 50 / 38
x_eks_1 = paralaksa_lin(faktor_paralakse_1, x_eks_1)
theta_1 = np.pi / 180 * 12
initial_eks_1 = x_eks_1[0], y_eks_1[0], np.average(vx_eks_1[0:3]), np.average(vy_eks_1[0:3])

# video_strmo
theta_strmo = np.pi / 180 * 19.19408 # kot se spreminja okoli 20; ce ga dam se v funkcional najde samo pri 19.19 minimum
t_eks_strmo, x_eks_strmo, y_eks_strmo, vx_eks_strmo, vy_eks_strmo = np.loadtxt('video_analiza_strmo.dat', skiprows=0, unpack=True, max_rows=None)
t_eks_strmo -= t_eks_strmo[0]
x_eks_strmo -= x_eks_strmo[0]
y_eks_strmo -= y_eks_strmo[0]
initial_eks_strmo = x_eks_strmo[0], y_eks_strmo[0], np.average(vx_eks_strmo[0:3]), np.average(vy_eks_strmo[0:3])

# video_2
theta_2 = np.pi / 180 * 14
t_eks_2, x_eks_2, y_eks_2, vx_eks_2, vy_eks_2 = np.loadtxt('video_analiza_2.dat', skiprows=0, unpack=True, max_rows=None)
t_eks_2 -= t_eks_2[0]
x_eks_2 -= x_eks_2[0]
y_eks_2 -= y_eks_2[0]
initial_eks_2 = x_eks_2[0], y_eks_2[0], np.average(vx_eks_2[0:3]), np.average(vy_eks_2[0:3])

# video_pocasi
theta_pocasi = np.pi / 180 * 16
t_eks_pocasi, x_eks_pocasi, y_eks_pocasi, vx_eks_pocasi, vy_eks_pocasi = np.loadtxt('video_analiza_pocasi.dat', skiprows=0, unpack=True, max_rows=None)
t_eks_pocasi -= t_eks_pocasi[0]
x_eks_pocasi -= x_eks_pocasi[0]
y_eks_pocasi -= y_eks_pocasi[0]
initial_eks_pocasi = x_eks_pocasi[0], y_eks_pocasi[0], np.average(vx_eks_pocasi[0:3]), np.average(vy_eks_pocasi[0:3])

# video_pocasi
theta_zelodesno = np.pi / 180 * 18
t_eks_zelodesno, x_eks_zelodesno, y_eks_zelodesno, vx_eks_zelodesno, vy_eks_zelodesno = np.loadtxt('video_analiza_zelodesno.dat', skiprows=0, unpack=True, max_rows=None)
t_eks_zelodesno -= t_eks_zelodesno[0]
x_eks_zelodesno -= x_eks_zelodesno[0]
y_eks_zelodesno -= y_eks_zelodesno[0]
faktor_paralakse_zelodesno = 37 / 28
x_eks_zelodesno = paralaksa_lin(faktor_paralakse_zelodesno, x_eks_zelodesno)
initial_eks_zelodesno = x_eks_zelodesno[0], y_eks_zelodesno[0], np.average(vx_eks_zelodesno[0:3]), np.average(vy_eks_zelodesno[0:3])
#################



# ####### video_1
# fig, ((ax1, axC1), (ax2, axC2), (ax3, axC3)) = plt.subplots(nrows=3, ncols=2, figsize = (10, 7))

# # clanek
# C_L1 = C_L_cutoff(0.188, 2.37, stall_angle)  # clanek
# C_D1 = C_D_cutoff(0.15, 1.24, C_90)
# plot_C_koef(axC1, 0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
# N_sistem1 = solution(t_eks_1, K, g, theta_1, C_L1, C_D1, stall_angle, initial_eks_1)[0]
# ax1.plot(N_sistem1[:, 0], N_sistem1[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax1.plot(x_eks_1, y_eks_1, '.', label='(x, y), eksperiment')
# ax1.grid(linestyle='--')
# ax1.legend(fancybox=False, prop={'size':8})
# ax1.set_xlabel('x [m]')
# ax1.set_ylabel('y [m]')
# ax1.axis('equal')
# ax1.set_title('Članek: C_L0, C_Lalpha, C_D0, C_Dalpha \n [0.188, 2.37, 0.15, 1.24]')

# # mthd2='Nelder-Mead'
# mthd2='TNC'
# # mthd2='L-BFGS-B'
# # mthd2='SLSQP'
# bnds2 = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# # bnds2 = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# # bnds2 = None
# weighted_1 = True
# C_fit2 = minimize(funkcional, (0.188, 2.37, 0.15, 1.24), \
#     args=(theta_1, t_eks_1, x_eks_1, y_eks_1, K, g, initial_eks_1, C_90, stall_angle, weighted_1), \
#          method=mthd2, bounds = bnds2, tol=1e-3)
# C_L2 = C_L_cutoff(C_fit2.x[0], C_fit2.x[1], stall_angle)  # 
# C_D2 = C_D_cutoff(C_fit2.x[2], C_fit2.x[3], C_90)
# plot_C_koef(axC2, C_fit2.x[0], C_fit2.x[1], stall_angle, C_fit2.x[2], C_fit2.x[3], C_90)
# N_sistem2 = solution(t_eks_1, K, g, theta_1, C_L2, C_D2, stall_angle, initial_eks_1)[0]
# ax2.plot(N_sistem2[:, 0], N_sistem2[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax2.plot(x_eks_1, y_eks_1, '.', label='(x, y), eksperiment')
# ax2.grid(linestyle='--')
# ax2.legend(fancybox=False, prop={'size':8})
# ax2.set_xlabel('x [m]')
# ax2.set_ylabel('y [m]')
# ax2.axis('equal')
# ax2.set_title('Minimization method={}, w={} \n {}'.format(mthd2, weighted_1, C_fit2.x))

# # mthd3='Nelder-Mead'
# mthd3='TNC'
# # mthd3='L-BFGS-B'
# # mthd3='SLSQP'
# bnds3 = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# # bnds3 = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# # bnds3 = None
# weighted_1 = False
# C_fit3 = minimize(funkcional, (0.188, 2.37, 0.15, 1.24), \
#      args=(theta_1, t_eks_1, x_eks_1, y_eks_1, K, g, initial_eks_1, C_90, stall_angle, weighted_1), \
#      method=mthd3, bounds = bnds3, tol=1e-3)
# C_L3 = C_L_cutoff(C_fit3.x[0], C_fit3.x[1], stall_angle)  # 
# C_D3 = C_D_cutoff(C_fit3.x[2], C_fit3.x[3], C_90)
# plot_C_koef(axC3, C_fit3.x[0], C_fit3.x[1], stall_angle, C_fit3.x[2], C_fit3.x[3], C_90)
# N_sistem3 = solution(t_eks_1, K, g, theta_1, C_L3, C_D3, stall_angle, initial_eks_1)[0]
# ax3.plot(N_sistem3[:, 0], N_sistem3[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax3.plot(x_eks_1, y_eks_1, '.', label='(x, y), eksperiment')
# ax3.grid(linestyle='--')
# ax3.legend(fancybox=False, prop={'size':8})
# ax3.set_xlabel('x [m]')
# ax3.set_ylabel('y [m]')
# ax3.axis('equal')
# ax3.set_title('Minimization method={}, w={} \n {}'.format(mthd3, weighted_1, C_fit3.x))

# fig.tight_layout()
# ###### konec video_1


####### video_strmo

# ## vse
# fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (8, 6))
# # ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
# # EKSPERIMENT #
# ax2.plot(x_eks_strmo, y_eks_strmo, '.', label='(x, y), eksperiment')
# ax3.plot(t_eks_strmo, x_eks_strmo, label='$x$')
# ax3.plot(t_eks_strmo, y_eks_strmo, label='$y$')
# ax3.plot(t_eks_strmo, vx_eks_strmo, label='$v_x$')
# ax3.plot(t_eks_strmo, vy_eks_strmo, label='$v_y$')
# ax3.grid(linestyle='--')
# ax3.legend(fancybox=False, prop={'size':9})
# ax3.set_title('Eksperiment')

# # mthd_strmo='Nelder-Mead'
# mthd_strmo='TNC'
# # mthd_strmo='L-BFGS-B'
# # mthd_strmo='SLSQP'
# # bnds3 = ((0.10, 0.22), (1.9, 2.7), (0.10, 0.19), (1.00, 1.40))
# # bnds3 = ((0.01, 1.), (0.01, 10.), (0.01, 10.), (0.01, 10.))
# # bnds_strmo = ((0.01, None), (0.01, None), (0.01, None), (0.01, None)) # zakaj ce ne omejim dela bolje
# bnds_strmo = None
# C_strmo = minimize(funkcional, (0.188, 2.37, 0.15, 1.24), \
#      args=(theta_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, K, g, initial_eks_strmo, C_90, stall_angle, False), \
#      method=mthd_strmo, bounds = bnds_strmo, tol=1e-3)
# C_Lstrmo = C_L_cutoff(C_strmo.x[0], C_strmo.x[1], stall_angle)  # 
# C_D3_strmo = C_D_cutoff(C_strmo.x[2], C_strmo.x[3], C_90)
# N_sistem_strmo = solution(t_eks_strmo, K, g, theta_strmo, C_Lstrmo, C_D3_strmo, stall_angle, initial_eks_strmo)[0]
# ax2.plot(N_sistem_strmo[:, 0], N_sistem_strmo[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax2.grid(linestyle='--')
# ax2.legend(fancybox=False, prop={'size':9})
# ax2.set_xlabel('x [m]')
# ax2.set_ylabel('y [m]')
# ax2.axis('equal')
# ax2.set_title('Trajektorija')
# plot_C_koef(ax4, C_strmo.x[0], C_strmo.x[1], stall_angle, C_strmo.x[2], C_strmo.x[3], C_90)

# ax1.plot(t_eks_strmo, N_sistem_strmo[:, 0], label='$x$')
# ax1.plot(t_eks_strmo, N_sistem_strmo[:, 1], label='$y$')
# ax1.plot(t_eks_strmo, N_sistem_strmo[:, 2], label='$v_x$')
# ax1.plot(t_eks_strmo, N_sistem_strmo[:, 3], label='$v_y$')
# ax1.grid(linestyle='--')
# ax1.legend(fancybox=False, prop={'size':9})
# ax1.set_title('Simulacija')
# fig.tight_layout()
# ## konec vse

# fig, ((ax1, axC1), (ax2, axC2), (ax3, axC3)) = plt.subplots(nrows=3, ncols=2, figsize = (10, 7))
# # clanek
# C_L1 = C_L_cutoff(0.188, 2.37, stall_angle)  # clanek
# C_D1 = C_D_cutoff(0.15, 1.24, C_90)
# plot_C_koef(axC1, 0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
# N_sistem1 = solution(t_eks_strmo, K, g, theta_strmo, C_L1, C_D1, stall_angle, initial_eks_strmo)[0]
# ax1.plot(N_sistem1[:, 0], N_sistem1[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax1.plot(x_eks_strmo, y_eks_strmo, '.', label='(x, y), eksperiment')
# ax1.grid(linestyle='--')
# ax1.legend(fancybox=False, prop={'size':8})
# ax1.set_xlabel('x [m]')
# ax1.set_ylabel('y [m]')
# ax1.axis('equal')
# ax1.set_title('Članek: C_L0, C_Lalpha, C_D0, C_Dalpha \n [0.188, 2.37, 0.15, 1.24]')

# mthd2='Nelder-Mead'
# # mthd2='TNC'
# # mthd2='L-BFGS-B'
# # mthd2='SLSQP'
# bnds2 = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# # bnds2 = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# # bnds2 = None
# weighted_strmo = True
# C_fit2 = minimize(funkcional, (0.188, 2.37, 0.15, 1.24), \
#     args=(theta_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, K, g, initial_eks_strmo, C_90, stall_angle, weighted_strmo), \
#          method=mthd2, bounds = bnds2, tol=1e-3)
# C_L2 = C_L_cutoff(C_fit2.x[0], C_fit2.x[1], stall_angle)
# C_D2 = C_D_cutoff(C_fit2.x[2], C_fit2.x[3], C_90)
# plot_C_koef(axC2, C_fit2.x[0], C_fit2.x[1], stall_angle, C_fit2.x[2], C_fit2.x[3], C_90)
# N_sistem2 = solution(t_eks_strmo, K, g, theta_strmo, C_L2, C_D2, stall_angle, initial_eks_strmo)[0]
# ax2.plot(N_sistem2[:, 0], N_sistem2[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax2.plot(x_eks_strmo, y_eks_strmo, '.', label='(x, y), eksperiment')
# ax2.grid(linestyle='--')
# ax2.legend(fancybox=False, prop={'size':8})
# ax2.set_xlabel('x [m]')
# ax2.set_ylabel('y [m]')
# ax2.axis('equal')
# ax2.set_title('Minimization method={}, w={} \n {}'.format(mthd2, weighted_strmo, C_fit2.x))

# mthd3='Nelder-Mead'
# # mthd3='TNC'
# # mthd3='L-BFGS-B'
# # mthd3='SLSQP'
# bnds3 = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# # bnds3 = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# # bnds3 = None
# weighted_strmo = False
# C_fit3 = minimize(funkcional, (0.188, 2.37, 0.15, 1.24), \
#      args=(theta_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, K, g, initial_eks_strmo, C_90, stall_angle, weighted_strmo), \
#      method=mthd3, bounds = bnds3, tol=1e-3)
# C_L3 = C_L_cutoff(C_fit3.x[0], C_fit3.x[1], stall_angle)  # 
# C_D3 = C_D_cutoff(C_fit3.x[2], C_fit3.x[3], C_90)
# plot_C_koef(axC3, C_fit3.x[0], C_fit3.x[1], stall_angle, C_fit3.x[2], C_fit3.x[3], C_90)
# N_sistem3 = solution(t_eks_strmo, K, g, theta_strmo, C_L3, C_D3, stall_angle, initial_eks_strmo)[0]
# ax3.plot(N_sistem3[:, 0], N_sistem3[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax3.plot(x_eks_strmo, y_eks_strmo, '.', label='(x, y), eksperiment')
# ax3.grid(linestyle='--')
# ax3.legend(fancybox=False, prop={'size':8})
# ax3.set_xlabel('x [m]')
# ax3.set_ylabel('y [m]')
# ax3.axis('equal')
# ax3.set_title('Minimization method={}, w={} \n {}'.format(mthd3, weighted_strmo, C_fit3.x))

# fig.tight_layout()
##### konec video_strmo


##### video_2
# ## vse
# fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (8, 6))
# # ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
# # EKSPERIMENT #
# ax2.plot(x_eks_2, y_eks_2, '.', label='(x, y), eksperiment')
# ax3.plot(t_eks_2, x_eks_2, label='$x$')
# ax3.plot(t_eks_2, y_eks_2, label='$y$')
# ax3.plot(t_eks_2, vx_eks_2, label='$v_x$')
# ax3.plot(t_eks_2, vy_eks_2, label='$v_y$')
# ax3.grid(linestyle='--')
# ax3.legend(fancybox=False, prop={'size':9})
# ax3.set_title('Eksperiment')

# # mthd_2='Nelder-Mead'
# mthd_2='TNC'
# # mthd_2='L-BFGS-B'
# # mthd_2='SLSQP'
# bnds_2 = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# # bnds_2 = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# # bnds_2 = None
# C_2 = minimize(funkcional, (0.188, 2.37, 0.15, 1.24), \
#      args=(theta_2, t_eks_2, x_eks_2, y_eks_2, K, g, initial_eks_2, C_90, stall_angle, False), \
#      method=mthd_2, bounds = bnds_2, tol=1e-3)
# C_L2 = C_L_cutoff(C_2.x[0], C_2.x[1], stall_angle)  # 
# C_D2 = C_D_cutoff(C_2.x[2], C_2.x[3], C_90)
# N_sistem_2 = solution(t_eks_2, K, g, theta_2, C_L2, C_D2, stall_angle, initial_eks_2)[0]
# ax2.plot(N_sistem_2[:, 0], N_sistem_2[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax2.grid(linestyle='--')
# ax2.legend(fancybox=False, prop={'size':9})
# ax2.set_xlabel('x [m]')
# ax2.set_ylabel('y [m]')
# ax2.axis('equal')
# ax2.set_title('Trajektorija')
# plot_C_koef(ax4, C_2.x[0], C_2.x[1], stall_angle, C_2.x[2], C_2.x[3], C_90)

# ax1.plot(t_eks_2, N_sistem_2[:, 0], label='$x$')
# ax1.plot(t_eks_2, N_sistem_2[:, 1], label='$y$')
# ax1.plot(t_eks_2, N_sistem_2[:, 2], label='$v_x$')
# ax1.plot(t_eks_2, N_sistem_2[:, 3], label='$v_y$')
# ax1.grid(linestyle='--')
# ax1.legend(fancybox=False, prop={'size':9})
# ax1.set_title('Simulacija')
# fig.tight_layout()
# # konec vse
# ##### konec video_2

# ###### video_zelodesno

# ## vse
# fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (8, 6))
# # ax2.plot(N_sistem[:, 0], N_sistem[:, 1], label='(x, y), simulacija, N sistem')
# # EKSPERIMENT #
# ax2.plot(x_eks_zelodesno, y_eks_zelodesno, '.', label='(x, y), eksperiment')
# ax3.plot(t_eks_zelodesno, x_eks_zelodesno, label='$x$')
# ax3.plot(t_eks_zelodesno, y_eks_zelodesno, label='$y$')
# ax3.plot(t_eks_zelodesno, vx_eks_zelodesno, label='$v_x$')
# ax3.plot(t_eks_zelodesno, vy_eks_zelodesno, label='$v_y$')
# ax3.grid(linestyle='--')
# ax3.legend(fancybox=False, prop={'size':9})
# ax3.set_title('Eksperiment')

# # mthd_zelodesno='Nelder-Mead'
# mthd_zelodesno='TNC'
# # mthd_zelodesno='L-BFGS-B'
# # mthd_zelodesno='SLSQP'
# # bnds3 = ((0.10, 0.22), (1.9, 2.7), (0.10, 0.19), (1.00, 1.40))
# # bnds3 = ((0.01, 1.), (0.01, 10.), (0.01, 10.), (0.01, 10.))
# # bnds_zelodesno = ((0.01, None), (0.01, None), (0.01, None), (0.01, None)) # zakaj ce ne omejim dela bolje
# bnds_zelodesno = None
# C_zelodesno = minimize(funkcional, (0.188, 2.37, 0.15, 1.24), \
#      args=(theta_zelodesno, t_eks_zelodesno, x_eks_zelodesno, y_eks_zelodesno, K, g, initial_eks_zelodesno, C_90, stall_angle, False), \
#      method=mthd_zelodesno, bounds = bnds_zelodesno, tol=1e-3)
# C_Lzelodesno = C_L_cutoff(C_zelodesno.x[0], C_zelodesno.x[1], stall_angle)  # 
# C_D3_zelodesno = C_D_cutoff(C_zelodesno.x[2], C_zelodesno.x[3], C_90)
# N_sistem_zelodesno = solution(t_eks_zelodesno, K, g, theta_zelodesno, C_Lzelodesno, C_D3_zelodesno, stall_angle, initial_eks_zelodesno)[0]
# ax2.plot(N_sistem_zelodesno[:, 0], N_sistem_zelodesno[:, 1], '.', label='(x, y), simulacija, N sistem')
# ax2.grid(linestyle='--')
# ax2.legend(fancybox=False, prop={'size':9})
# ax2.set_xlabel('x [m]')
# ax2.set_ylabel('y [m]')
# ax2.axis('equal')
# ax2.set_title('Trajektorija')
# plot_C_koef(ax4, C_zelodesno.x[0], C_zelodesno.x[1], stall_angle, C_zelodesno.x[2], C_zelodesno.x[3], C_90)

# ax1.plot(t_eks_zelodesno, N_sistem_zelodesno[:, 0], label='$x$')
# ax1.plot(t_eks_zelodesno, N_sistem_zelodesno[:, 1], label='$y$')
# ax1.plot(t_eks_zelodesno, N_sistem_zelodesno[:, 2], label='$v_x$')
# ax1.plot(t_eks_zelodesno, N_sistem_zelodesno[:, 3], label='$v_y$')
# ax1.grid(linestyle='--')
# ax1.legend(fancybox=False, prop={'size':9})
# ax1.set_title('Simulacija')
# fig.tight_layout()
# ## konec zelodesno


def funkcional_skupaj(x, lst, K, g, C_90, stall_angle, weighted):
    C_L0, C_Lalpha, C_D0, C_Dalpha = x
    C_L = C_L_cutoff(C_L0, C_Lalpha, stall_angle)
    C_D = C_D_cutoff(C_D0, C_Dalpha, C_90)
    distance = 0.
    for e in lst:
        theta, t_eks, x_eks, y_eks, initial_eks = e
        N_sistem = solution(t_eks, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
    
        if weighted:
            weights = np.exp(-t_eks * 2.)
        else:
            weights = 1.

        distance += np.average(weights * ((N_sistem[:, 0] - x_eks)**2 + (N_sistem[:, 1] - y_eks)**2))
    return np.average(distance)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize = (10, 9))
plot_C_koef(ax1, 0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
ax1.set_title('C koeficienta: članek')
# mthd='Nelder-Mead'
mthd='TNC'
# mthd='L-BFGS-B'
# mthd='SLSQP'
bnds = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# bnds = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# bnds = None
weighted = True
C_fit = minimize(funkcional_skupaj, (0.188, 2.37, 0.15, 1.24), \
    args=([ (theta_1, t_eks_1, x_eks_1, y_eks_1, initial_eks_1),\
         (theta_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, initial_eks_strmo),\
         (theta_2, t_eks_2, x_eks_2, y_eks_2, initial_eks_2), \
         (theta_pocasi, t_eks_pocasi, x_eks_pocasi, y_eks_pocasi, initial_eks_pocasi), \
         (theta_zelodesno, t_eks_zelodesno, x_eks_zelodesno, y_eks_zelodesno, initial_eks_zelodesno)], \
         K, g, C_90, stall_angle, weighted), \
        method=mthd, bounds = bnds, tol=1e-3)
C_L = C_L_cutoff(C_fit.x[0], C_fit.x[1], stall_angle)  # 
C_D = C_D_cutoff(C_fit.x[2], C_fit.x[3], C_90)
plot_C_koef(ax2, C_fit.x[0], C_fit.x[1], stall_angle, C_fit.x[2], C_fit.x[3], C_90)
ax2.set_title('C koeficienta: minimizacija')
N_sistem_1 = solution(t_eks_1, K, g, theta_1, C_L, C_D, stall_angle, initial_eks_1)[0]
N_sistem_strmo = solution(t_eks_strmo, K, g, theta_strmo, C_L, C_D, stall_angle, initial_eks_strmo)[0]
N_sistem_2 = solution(t_eks_2, K, g, theta_2, C_L, C_D, stall_angle, initial_eks_2)[0]
N_sistem_pocasi = solution(t_eks_pocasi, K, g, theta_pocasi, C_L, C_D, stall_angle, initial_eks_pocasi)[0]
N_sistem_zelodesno = solution(t_eks_zelodesno, K, g, theta_zelodesno, C_L, C_D, stall_angle, initial_eks_zelodesno)[0]

ax3.plot(N_sistem_1[:, 0], N_sistem_1[:, 1], '.', label='video_1 simulacija')
ax3.plot(x_eks_1, y_eks_1, '.', label='video_1, eksperiment')
ax3.grid(linestyle='--')
ax3.legend(fancybox=False, prop={'size':8})
ax3.set_xlabel('x [m]')
ax3.set_ylabel('y [m]')
ax3.axis('equal')
ax3.set_title('Minimization method={}, w={} \n {}'.format(mthd, weighted, C_fit.x))
ax4.plot(N_sistem_strmo[:, 0], N_sistem_strmo[:, 1], '.', label='video_strmo simulacija')
ax4.plot(x_eks_strmo, y_eks_strmo, '.', label='video_strmo, eksperiment')
ax4.grid(linestyle='--')
ax4.legend(fancybox=False, prop={'size':8})
ax4.set_xlabel('x [m]')
ax4.set_ylabel('y [m]')
ax4.axis('equal')
ax4.set_title('Minimization method={}, w={} \n {}'.format(mthd, weighted, C_fit.x))
ax5.plot(N_sistem_2[:, 0], N_sistem_2[:, 1], '.', label='video_2 simulacija')
ax5.plot(x_eks_2, y_eks_2, '.', label='video_2, eksperiment')
ax5.grid(linestyle='--')
ax5.legend(fancybox=False, prop={'size':8})
ax5.set_xlabel('x [m]')
ax5.set_ylabel('y [m]')
ax5.axis('equal')
ax5.set_title('Minimization method={}, w={} \n {}'.format(mthd, weighted, C_fit.x))
ax6.plot(N_sistem_pocasi[:, 0], N_sistem_pocasi[:, 1], '.', label='video_pocasi simulacija')
ax6.plot(x_eks_pocasi, y_eks_pocasi, '.', label='video_pocasi, eksperiment')
ax6.grid(linestyle='--')
ax6.legend(fancybox=False, prop={'size':8})
ax6.set_xlabel('x [m]')
ax6.set_ylabel('y [m]')
ax6.axis('equal')
ax6.set_title('Minimization method={}, w={} \n {}'.format(mthd, weighted, C_fit.x))
ax7.plot(N_sistem_zelodesno[:, 0], N_sistem_zelodesno[:, 1], '.', label='video_zelodesno simulacija')
ax7.plot(x_eks_zelodesno, y_eks_zelodesno, '.', label='video_zelodesno, eksperiment')
ax7.grid(linestyle='--')
ax7.legend(fancybox=False, prop={'size':8})
ax7.set_xlabel('x [m]')
ax7.set_ylabel('y [m]')
ax7.axis('equal')
ax7.set_title('Minimization method={}, w={} \n {}'.format(mthd, weighted, C_fit.x))

fig.tight_layout()

plt.show()