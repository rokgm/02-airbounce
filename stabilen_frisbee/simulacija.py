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
    axes.set_xlabel('angle of attack [$^\circ$]', fontsize=13)
    axes.legend()
    axes.grid(linestyle='--')
    axes.set_title('Lift and drag coefficient', fontsize=17)

def funkcional(x, t_eks1, x_eks1, y_eks1, K, g, initial_eks, C_90, stall_angle, weighted):
    theta_ef, C_L0, C_Lalpha, C_D0, C_Dalpha= x
    C_L = C_L_cutoff(C_L0, C_Lalpha, stall_angle)
    C_D = C_D_cutoff(C_D0, C_Dalpha, C_90)
    N_sistem = solution(t_eks1, K, g, theta_ef, C_L, C_D, stall_angle, initial_eks)[0]
    
    if weighted:
        #weights = np.exp(-t_eks1 * 3.)
        weights = 1 / (0.05 + t_eks1**2 + 0.01)**2
    else:
        weights = 1.

    distance = weights * ((N_sistem[:, 0] - x_eks1)**2 + (N_sistem[:, 1] - y_eks1)**2)
    return np.average(distance)

def funkcional_skupaj(x, lst, K, g, C_90, stall_angle, weighted):
    C_L0, C_Lalpha, C_D0, C_Dalpha = x
    C_L = C_L_cutoff(C_L0, C_Lalpha, stall_angle)
    C_D = C_D_cutoff(C_D0, C_Dalpha, C_90)
    distance = 0.
    for e in lst:
        theta, t_eks, x_eks, y_eks, initial_eks = e
        N_sistem = solution(t_eks, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
    
        if weighted:
            #weights = np.exp(-t_eks * 2.)
            weights = 1 / (0.05 + t_eks**2 + 0.01)**2
        else:
            weights = 1.

        distance += np.average(weights * ((N_sistem[:, 0] - x_eks)**2 + (N_sistem[:, 1] - y_eks)**2))
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

## Podatki ##     paralksa ni upostevana pri hitrostih, za zacetne pogoje nima veliko vpliva
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

# video_zelodesno
theta_zelodesno = np.pi / 180 * 18
t_eks_zelodesno, x_eks_zelodesno, y_eks_zelodesno, vx_eks_zelodesno, vy_eks_zelodesno = np.loadtxt('video_analiza_zelodesno.dat', skiprows=0, unpack=True, max_rows=None)
t_eks_zelodesno -= t_eks_zelodesno[0]
x_eks_zelodesno -= x_eks_zelodesno[0]
y_eks_zelodesno -= y_eks_zelodesno[0]
faktor_paralakse_zelodesno = 37 / 28
x_eks_zelodesno = paralaksa_lin(faktor_paralakse_zelodesno, x_eks_zelodesno)
initial_eks_zelodesno = x_eks_zelodesno[0], y_eks_zelodesno[0], np.average(vx_eks_zelodesno[0:3]), np.average(vy_eks_zelodesno[0:3])

# video_3
theta_3 = np.pi / 180 * 10
t_eks_3, x_eks_3, y_eks_3, vx_eks_3, vy_eks_3 = np.loadtxt('video_analiza_3.dat', skiprows=0, unpack=True, max_rows=None)
t_eks_3 -= t_eks_3[0]
x_eks_3 -= x_eks_3[0]
y_eks_3 -= y_eks_3[0]
initial_eks_3 = x_eks_3[0], y_eks_3[0], np.average(vx_eks_3[0:3]), np.average(vy_eks_3[0:3])
#################

def plot_all(theta, t_eks, x_eks, y_eks, vx_eks, vy_eks, initial_eks, C, C_L, C_D):
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (8, 6))
    # EKSPERIMENT #
    ax2.plot(x_eks, y_eks, '.', label='experiment')
    ax3.plot(t_eks, x_eks, label='$x$')
    ax3.plot(t_eks, y_eks, label='$y$')
    ax3.plot(t_eks, vx_eks, label='$v_x$')
    ax3.plot(t_eks, vy_eks, label='$v_y$')
    ax3.grid(linestyle='--')
    ax3.set_xlabel('t [s]', fontsize=13)
    ax3.set_ylabel('[m, m/s]', fontsize=13)
    ax3.legend(fancybox=False, prop={'size':9})
    ax3.set_title('Experiment', fontsize=17)

    N_sistem = solution(t_eks, K, g, theta, C_L, C_D, stall_angle, initial_eks)[0]
    ax2.plot(N_sistem[:, 0], N_sistem[:, 1], '.', label='simulation')
    ax2.grid(linestyle='--')
    ax2.legend(fancybox=False, prop={'size':9})
    ax2.set_xlabel('x [m]', fontsize=13)
    ax2.set_ylabel('y [m]', fontsize=13)
    #ax2.axis('equal')
    ax2.set_title('Trajectory', fontsize=17)
    #plot_C_koef(ax4, C.x[1], C.x[2], stall_angle, C.x[3], C.x[4], C_90)
    plot_C_koef(ax4, 0.188, 2.37, stall_angle,  0.15, 1.24, C_90)

    ax1.plot(t_eks, N_sistem[:, 0], label='$x$')
    ax1.plot(t_eks, N_sistem[:, 1], label='$y$')
    ax1.plot(t_eks, N_sistem[:, 2], label='$v_x$')
    ax1.plot(t_eks, N_sistem[:, 3], label='$v_y$')
    ax1.grid(linestyle='--')
    ax1.legend(fancybox=False, prop={'size':9})
    ax1.set_title('Simulation', fontsize=17)
    ax1.set_xlabel('t [s]', fontsize=13)
    ax1.set_ylabel('[m, m/s]', fontsize=13)
    fig.tight_layout()
    # fig.savefig(path_predstavitev + '/strmo_clanek_vse.pdf')

def minimizacija(theta, t_eks, x_eks, y_eks, initial_eks):
    mthd='TNC'
    weighted=True
    #bnds = ((0.16, 0.55), (0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
    bnds = ((0.16, 0.55), (0.12, 0.20), (2.0, 2.6), (0.18, 0.26), (0.8, 1.4))
    C = minimize(funkcional, (theta, 0.188, 2.37, 0.15, 1.24), \
        args=(t_eks, x_eks, y_eks, K, g, initial_eks, C_90, stall_angle, weighted), \
        method=mthd, bounds=bnds, tol=1e-3)
    C_L = C_L_cutoff(C.x[1], C.x[2], stall_angle)
    C_D = C_D_cutoff(C.x[3], C.x[4], C_90)
    theta_ef = C.x[0]
    return theta_ef, C, C_L, C_D

path_predstavitev = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'predstavitev'))

C_L_strmo = C_L_cutoff(0.188, 2.37, stall_angle)
C_D_strmo= C_D_cutoff(0.15, 1.24, C_90)
# plot_all(theta_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, vx_eks_strmo, vy_eks_strmo, initial_eks_strmo, [None, 0.188, 2.37, 0.15, 1.24], C_L_strmo, C_D_strmo)

# omejeno na skupni fit
theta_ef_1, C_1, C_L_1, C_D_1 = minimizacija(theta_1, t_eks_1, x_eks_1, y_eks_1, initial_eks_1)
print(theta_ef_1 - theta_1)

theta_ef_strmo, C_strmo, C_L_strmo, C_D_strmo = minimizacija(theta_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, initial_eks_strmo)
print(theta_ef_strmo - theta_strmo)

theta_ef_2, C_2, C_L_2, C_D_2 = minimizacija(theta_2, t_eks_2, x_eks_2, y_eks_2, initial_eks_2)
print(theta_ef_2 - theta_2)

theta_ef_pocasi, C_pocasi, C_L_pocasi, C_D_pocasi = minimizacija(theta_pocasi, t_eks_pocasi, x_eks_pocasi, y_eks_pocasi, initial_eks_pocasi)
print(theta_ef_pocasi - theta_pocasi)

theta_ef_zelodesno, C_zelodesno, C_L_zelodesno, C_D_zelodesno = minimizacija(theta_zelodesno, t_eks_zelodesno, x_eks_zelodesno, y_eks_zelodesno, initial_eks_zelodesno)
print(theta_ef_zelodesno - theta_zelodesno)

theta_ef_3, C_3, C_L_3, C_D_3 = minimizacija(theta_3, t_eks_3, x_eks_3, y_eks_3, initial_eks_3)
print(theta_ef_3 - theta_3)

fig, ((ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=3, ncols=2, figsize = (8, 6))
fig1, (axC1, axC2) = plt.subplots(nrows=2, ncols=1, figsize = (5, 5))
plot_C_koef(axC1, 0.188, 2.37, stall_angle, 0.15, 1.24, C_90)
axC1.set_title('article [1]')
# mthd='Nelder-Mead'
mthd='TNC'
# mthd='L-BFGS-B'
# mthd='SLSQP'
bnds = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# bnds = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# bnds = None
weighted = True
C_fit = minimize(funkcional_skupaj, (0.188, 2.37, 0.15, 1.24), \
    args=([ (theta_ef_1, t_eks_1, x_eks_1, y_eks_1, initial_eks_1),\
         (theta_ef_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, initial_eks_strmo),\
         (theta_ef_2, t_eks_2, x_eks_2, y_eks_2, initial_eks_2), \
         (theta_ef_pocasi, t_eks_pocasi, x_eks_pocasi, y_eks_pocasi, initial_eks_pocasi), \
         (theta_ef_zelodesno, t_eks_zelodesno, x_eks_zelodesno, y_eks_zelodesno, initial_eks_zelodesno)], \
         K, g, C_90, stall_angle, weighted), \
        method=mthd, bounds = bnds, tol=1e-3)
C_L = C_L_cutoff(C_fit.x[0], C_fit.x[1], stall_angle)
C_D = C_D_cutoff(C_fit.x[2], C_fit.x[3], C_90)
plot_C_koef(axC2, C_fit.x[0], C_fit.x[1], stall_angle, C_fit.x[2], C_fit.x[3], C_90)
axC2.set_title('weighted minimization')
fig1.suptitle('Lift and Drag Coefficients', fontsize=16)
print('cji:', C_fit.x[0], C_fit.x[1], C_fit.x[2], C_fit.x[3])

# C_L = C_L_cutoff(0.188, 2.37, stall_angle)
# C_D = C_D_cutoff(0.15, 1.24, C_90)
N_sistem_1 = solution(t_eks_1, K, g, theta_ef_1, C_L, C_D, stall_angle, initial_eks_1)[0]
N_sistem_strmo = solution(t_eks_strmo, K, g, theta_ef_strmo, C_L, C_D, stall_angle, initial_eks_strmo)[0]
N_sistem_2 = solution(t_eks_2, K, g, theta_ef_2, C_L, C_D, stall_angle, initial_eks_2)[0]
N_sistem_pocasi = solution(t_eks_pocasi, K, g, theta_ef_pocasi, C_L, C_D, stall_angle, initial_eks_pocasi)[0]
N_sistem_zelodesno = solution(t_eks_zelodesno, K, g, theta_ef_zelodesno, C_L, C_D, stall_angle, initial_eks_zelodesno)[0]

ax3.plot(N_sistem_1[:, 0], N_sistem_1[:, 1], '.', label='sim')
ax3.plot(x_eks_1, y_eks_1, '.', label='exp')
ax3.grid(linestyle='--')
ax3.legend(fancybox=False, prop={'size':8})
ax3.set_xlabel('x [m]')
ax3.set_ylabel('y [m]')
# ax3.axis('equal')
ax3.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_1 * 180 / np.pi, initial_eks_1[2], initial_eks_1[3]))
ax7.plot(N_sistem_strmo[:, 0], N_sistem_strmo[:, 1], '.', label='sim')
ax7.plot(x_eks_strmo, y_eks_strmo, '.', label='exp')
ax7.grid(linestyle='--')
ax7.legend(fancybox=False, prop={'size':8})
ax7.set_xlabel('x [m]')
ax7.set_ylabel('y [m]')
# ax7.axis('equal')
ax7.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_strmo * 180 / np.pi, initial_eks_strmo[2], initial_eks_strmo[3]))
ax5.plot(N_sistem_2[:, 0], N_sistem_2[:, 1], '.', label='sim')
ax5.plot(x_eks_2, y_eks_2, '.', label='exp')
ax5.grid(linestyle='--')
ax5.legend(fancybox=False, prop={'size':8})
ax5.set_xlabel('x [m]')
ax5.set_ylabel('y [m]')
# ax5.axis('equal')
ax5.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_2 * 180 / np.pi, initial_eks_2[2], initial_eks_2[3]))
ax6.plot(N_sistem_pocasi[:, 0], N_sistem_pocasi[:, 1], '.', label='sim')
ax6.plot(x_eks_pocasi, y_eks_pocasi, '.', label='exp')
ax6.grid(linestyle='--')
ax6.legend(fancybox=False, prop={'size':8})
ax6.set_xlabel('x [m]')
ax6.set_ylabel('y [m]')
# ax6.axis('equal')
ax6.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_pocasi * 180 / np.pi, initial_eks_pocasi[2], initial_eks_pocasi[3]))
ax4.plot(N_sistem_zelodesno[:, 0], N_sistem_zelodesno[:, 1], '.', label='sim')
ax4.plot(x_eks_zelodesno, y_eks_zelodesno, '.', label='exp')
ax4.grid(linestyle='--')
ax4.legend(fancybox=False, prop={'size':8})
ax4.set_xlabel('x [m]')
ax4.set_ylabel('y [m]')
# ax4.axis('equal')
ax4.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_zelodesno * 180 / np.pi, initial_eks_zelodesno[2], initial_eks_zelodesno[3]))

N_sistem_3 = solution(t_eks_3, K, g, theta_3, C_L, C_D, stall_angle, initial_eks_3)[0]
ax8.plot(N_sistem_3[:, 0], N_sistem_3[:, 1], '.', label='sim')
ax8.plot(x_eks_3, y_eks_3, '.', label='exp')
ax8.grid(linestyle='--')
ax8.legend(fancybox=False, prop={'size':8})
ax8.set_xlabel('x [m]')
ax8.set_ylabel('y [m]')
# ax8.axis('equal')
ax8.set_title('TEST, EXCLUDED FROM FIT \n' + r'$\theta = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_3 * 180 / np.pi, initial_eks_zelodesno[2], initial_eks_zelodesno[3]))
#fig.delaxes(ax8)

fig.suptitle('Trajectories: Weighted Minimization and Experiment', fontsize=16)
fig1.tight_layout()
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

# fig.savefig(path_predstavitev + '/traj_weighted_mini.pdf')
# fig1.savefig(path_predstavitev + '/cji_clanek_weighted.pdf')


fig_un, ((ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=3, ncols=2, figsize = (8, 6))
fig1_un, (axC1, axC2) = plt.subplots(nrows=2, ncols=1, figsize = (5, 5))
plot_C_koef(axC1, C_fit.x[0], C_fit.x[1], stall_angle, C_fit.x[2], C_fit.x[3], C_90)
axC1.set_title('weighted minimization')
# mthd='Nelder-Mead'
mthd='TNC'
# mthd='L-BFGS-B'
# mthd='SLSQP'
bnds = ((0.01, 0.3), (0.01, 3.), (0.01, 0.3), (0.01, 2.))
# bnds = ((0.01, None), (0.01, None), (0.01, None), (0.01, None))
# bnds = None
weighted = False
C_fit_un = minimize(funkcional_skupaj, (0.188, 2.37, 0.15, 1.24), \
    args=([ (theta_ef_1, t_eks_1, x_eks_1, y_eks_1, initial_eks_1),\
         #(theta_ef_strmo, t_eks_strmo, x_eks_strmo, y_eks_strmo, initial_eks_strmo),\
         (theta_ef_2, t_eks_2, x_eks_2, y_eks_2, initial_eks_2), \
         (theta_ef_pocasi, t_eks_pocasi, x_eks_pocasi, y_eks_pocasi, initial_eks_pocasi), \
         (theta_ef_zelodesno, t_eks_zelodesno, x_eks_zelodesno, y_eks_zelodesno, initial_eks_zelodesno)], \
         K, g, C_90, stall_angle, weighted), \
        method=mthd, bounds = bnds, tol=1e-3)
C_L = C_L_cutoff(C_fit_un.x[0], C_fit_un.x[1], stall_angle)
C_D = C_D_cutoff(C_fit_un.x[2], C_fit_un.x[3], C_90)
plot_C_koef(axC2, C_fit_un.x[0], C_fit_un.x[1], stall_angle, C_fit_un.x[2], C_fit_un.x[3], C_90)
axC2.set_title('unweighted minimization')
fig1_un.suptitle('Lift and Drag Coefficients', fontsize=16)
print('cji:', C_fit_un.x[0], C_fit_un.x[1], C_fit_un.x[2], C_fit_un.x[3])

# C_L = C_L_cutoff(0.188, 2.37, stall_angle)
# C_D = C_D_cutoff(0.15, 1.24, C_90)
N_sistem_1_un = solution(t_eks_1, K, g, theta_ef_1, C_L, C_D, stall_angle, initial_eks_1)[0]
N_sistem_strmo_un = solution(t_eks_strmo, K, g, theta_ef_strmo, C_L, C_D, stall_angle, initial_eks_strmo)[0]
N_sistem_2_un = solution(t_eks_2, K, g, theta_ef_2, C_L, C_D, stall_angle, initial_eks_2)[0]
N_sistem_pocasi_un = solution(t_eks_pocasi, K, g, theta_ef_pocasi, C_L, C_D, stall_angle, initial_eks_pocasi)[0]
N_sistem_zelodesno_un = solution(t_eks_zelodesno, K, g, theta_ef_zelodesno, C_L, C_D, stall_angle, initial_eks_zelodesno)[0]

ax3.plot(N_sistem_1_un[:, 0], N_sistem_1_un[:, 1], '.', label='unw')
ax3.plot(N_sistem_1[:, 0], N_sistem_1[:, 1], '.', label='w')
ax3.grid(linestyle='--')
ax3.legend(fancybox=False, prop={'size':8})
ax3.set_xlabel('x [m]')
ax3.set_ylabel('y [m]')
# ax3.axis('equal')
ax3.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_1 * 180 / np.pi, initial_eks_1[2], initial_eks_1[3]))
ax7.plot(N_sistem_strmo_un[:, 0], N_sistem_strmo_un[:, 1], '.', label='unw')
ax7.plot(N_sistem_strmo[:, 0], N_sistem_strmo[:, 1], '.', label='w')
ax7.grid(linestyle='--')
ax7.legend(fancybox=False, prop={'size':8})
ax7.set_xlabel('x [m]')
ax7.set_ylabel('y [m]')
# ax7.axis('equal')
ax7.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_strmo * 180 / np.pi, initial_eks_strmo[2], initial_eks_strmo[3]))
ax5.plot(N_sistem_2_un[:, 0], N_sistem_2_un[:, 1], '.', label='unw')
ax5.plot(N_sistem_2[:, 0], N_sistem_2[:, 1], '.', label='w')
ax5.grid(linestyle='--')
ax5.legend(fancybox=False, prop={'size':8})
ax5.set_xlabel('x [m]')
ax5.set_ylabel('y [m]')
# ax5.axis('equal')
ax5.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_2 * 180 / np.pi, initial_eks_2[2], initial_eks_2[3]))
ax6.plot(N_sistem_pocasi_un[:, 0], N_sistem_pocasi_un[:, 1], '.', label='unw')
ax6.plot(N_sistem_pocasi[:, 0], N_sistem_pocasi[:, 1], '.', label='w')
ax6.grid(linestyle='--')
ax6.legend(fancybox=False, prop={'size':8})
ax6.set_xlabel('x [m]')
ax6.set_ylabel('y [m]')
# ax6.axis('equal')
ax6.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_pocasi * 180 / np.pi, initial_eks_pocasi[2], initial_eks_pocasi[3]))
ax4.plot(N_sistem_zelodesno_un[:, 0], N_sistem_zelodesno_un[:, 1], '.', label='unw')
ax4.plot(N_sistem_zelodesno[:, 0], N_sistem_zelodesno[:, 1], '.', label='w')
ax4.grid(linestyle='--')
ax4.legend(fancybox=False, prop={'size':8})
ax4.set_xlabel('x [m]')
ax4.set_ylabel('y [m]')
# ax4.axis('equal')
ax4.set_title(r'$\theta_{{ef}} = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_ef_zelodesno * 180 / np.pi, initial_eks_zelodesno[2], initial_eks_zelodesno[3]))

N_sistem_3_un = solution(t_eks_3, K, g, theta_3, C_L, C_D, stall_angle, initial_eks_3)[0]
ax8.plot(N_sistem_3_un[:, 0], N_sistem_3_un[:, 1], '.', label='unw')
ax8.plot(N_sistem_3[:, 0], N_sistem_3[:, 1], '.', label='w')
ax8.grid(linestyle='--')
ax8.legend(fancybox=False, prop={'size':8})
ax8.set_xlabel('x [m]')
ax8.set_ylabel('y [m]')
# ax8.axis('equal')
ax8.set_title('TEST, EXCLUDED FROM FIT \n' + r'$\theta = {:.0f}^{{\circ}}$, $v_x = {:.2f}$ m/s, $v_y = {:.2f}$ m/s'.format(theta_3 * 180 / np.pi, initial_eks_zelodesno[2], initial_eks_zelodesno[3]))
#fig_un.delaxes(ax8)

fig_un.suptitle('Minimization: Weighted vs Unweighted', fontsize=16)
fig1_un.tight_layout()
fig_un.tight_layout(rect=[0, 0.03, 1, 0.95])
fig1_un.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig_un.savefig(path_predstavitev + '/traj_unweighted_weighted.pdf')
# fig1_un.savefig(path_predstavitev + '/cji_weighted_unweighted.pdf')

plt.show()