# from mpl_toolkits.mplot3d import axes3d
import random

import matplotlib.animation as animation
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.constants import g, mu_0

############################ CONSTANTS and FUNCTIONS ############################
L = 0.18  # visina magneta na kateri visi
h = 0.02  # višina med ravnino magnetov ter nihalom pod theta = 0
M = 1/1000
mag_moment = 1

# INITIAL CONDITIONS
v = 1.4
phi = 2*np.pi*np.random.random()
x0 = 0.
y0 = 0.
v_x0 = v * np.cos(phi)
v_y0 = v * np.sin(phi)
initial_conditions = [x0, y0, v_x0, v_y0]

dx = 0.05
dy = 0.05
border = L
x = np.arange(-border, border + dx, dx)
y = np.arange(-border, border + dy, dy)

m1 = np.array([0, 0, mag_moment])  # magnetic moment
m2 = np.array([0, 0, mag_moment])

dt = dx
t = np.arange(0.0, 10*len(x) * dt, dt)


# magnetic potencial
def A(r, m):
    return mu_0/4*np.pi * (np.cross(m, r) / r**3)

# magnetic field


def B(r, m):
    return mu_0/4*np.pi * (3. * r * np.dot(m, r)/np.linalg.norm(r)**5 - m/np.linalg.norm(r)**3)

# magnetic force between two magnetic dipoles


def F(r, m1, m2):
    return 3*mu_0 / (4*np.pi*np.linalg.norm(r)**5) * ((np.dot(m1, r)*m2 + np.dot(m2, r)*m1 + np.dot(m1, m2)*r - 5*np.dot(m1, r)*np.dot(m2, r) / np.linalg.norm(r)**2 * r))

# magnetic force between two cylindrical magnets
# def F(r):
#     return np.pi*mu_0/4 *


############################  POSITION OF MAGNETS  ############################
def configuration_of_magnets(x, y, N, distance=1, random_positions=False):
    if random_positions == False:
        theta = 2*np.pi / N
        phi = 0
        dx = np.abs((x[1] - x[0]))
        dy = np.abs((y[1] - y[0]))
        multiplication_factor = distance / dx
        positions = np.zeros([N, 2])
        # positions[0] = [0, 0]
        for i in range(1, N):
            x_projection = multiplication_factor * np.cos(phi)
            y_projection = multiplication_factor * np.sin(phi)
            x_direction = x_projection * dx
            y_direction = y_projection * dy

            positions[i] = positions[i - 1][0] + \
                x_direction, positions[i - 1][1] + y_direction
            phi += theta

        centroid = positions.mean(axis=0)
        move_in_x = centroid[0]
        move_in_y = centroid[1]

        final_positions = np.zeros([N, 2])
        for i in range(N):
            final_positions[i] = positions[i][0] - \
                move_in_x, positions[i][1] - move_in_y
        return final_positions

    else:
        positions = np.zeros([N, 2])
        for i in range(N):
            theta = 2 * np.pi * random.random()
            lenght = random.random()
            positions[i] = lenght*np.cos(theta), lenght*np.sin(theta)
        return positions


def magnetic_field(x, y, number_of_magnets):
    N = len(x)
    M = len(y)
    positions = configuration_of_magnets(x, y, number_of_magnets)
    mag_field = np.zeros([N, M])
    for magnet in positions:
        for i in range(N):
            for j in range(M):
                r = np.array([magnet[0] - x[i], magnet[1] - y[j], 0])
                mag_field[i, j] += np.linalg.norm(B(r, m1))
    return mag_field, positions


# mag_field = magnetic_field(x, y, 6)[0]
# position_of_magnets = magnetic_field(x, y, 3)[1]


############################  FORCE and PENDULUM STATE  ############################
def force(x, y, magnets):
    # x = position[0]
    # y = position[1]
    # z = position[2]
    combined_force_of_magnets = np.zeros(3)
    pendulum_force = np.zeros(2)
    r = (x**2 + y**2)**0.5
    theta = np.arctan(r/L)
    # alpha = np.arctan(y/x)
    for i in range(len(magnets)):
        r_vec = np.array([(x - magnets[i][0]),
                         (y - magnets[i][1]), (h + L - np.sqrt(L**2 - x**2 - y**2))])
        force_ = F(r_vec, m1, m2)

        combined_force_of_magnets[0] += force_[0]
        combined_force_of_magnets[1] += force_[1]
        combined_force_of_magnets[2] += force_[2]
        # print("vektor z predznak", np.sign(
        #print("sila predznak", np.sign(force_[2]))
    # pendulum_force[0] = omega**2 * np.sin(theta) * x/r
    # pendulum_force[1] = omega**2 * np.sin(theta) * y/r
    gravity = [0, 0, - g*M]
    force = gravity + combined_force_of_magnets
    vec = [x, y, -L - h + np.sqrt(L**2 - x**2 - y**2)]
    vec = [np.dot(force, vec) * vec[0], np.dot(force, vec)
           * vec[1], np.dot(force, vec) * vec[2]]
    force = force - vec
    # I AM NOT SURE ABOUT SIGNS +/-
    return force[: -1]


# a simple pendulum y''= F(y) , state = (y,v)
def pendulum(state, t, magnets):
    x, y, v_x, v_y = state
    derivitives = np.zeros_like(state)
    derivitives[0] = v_x  # dxdt[0] = derivites[0] = state[1] = v_x
    derivitives[1] = v_y  # dydt[0] = derivites[1] = state[2] = v_y
    derivitives[2] = force(x, y, magnets)[0]  # dxdt[1] = derivites[2] = F_x
    derivitives[3] = force(x, y, magnets)[1]  # dydt[1] = derivites[3] = F_y
    # dydt[0] = state[1][1] # x' = v
    # dydt[1] = force(state[1][0])  # v' = F(x)
    return derivitives


magnets = configuration_of_magnets(x, y, 5, 0.5, False)
# print(magnets[1])
solution = odeint(pendulum, initial_conditions, t, args=(magnets,))

# plt.plot(magnets[:, 0], magnets[:, 1], "o")
# # plt.plot(magnets[1][0], magnets[1][1], "o")
# # plt.xlim(-3, 3)
# # plt.ylim(-3, 3)
# plt.grid()
# plt.show()
#
# plt.plot(t, solution[:, 0], label="x")  # resitev za x
# plt.plot(t, solution[:, 1], label="y")  # resitev za y
# plt.plot(t, solution[:, 2])
# plt.legend()
# plt.show()

# for i in [0, 10, 20, 50, 100, 120, 150, 175, 200]:
#     plt.plot(solution[i, 0], solution[i, 1], "o", label="{}s".format(i))
# plt.plot(solution[175, 0], solution[175, 1], "o")
# plt.plot(magnets[:, 0], magnets[:, 1], "o",
#          markersize=10, color="black", label="magnets")
# # plt.xlim(-border - 1, border + 1)
# # plt.ylim(-border - 1, border + 1)
# plt.grid()
# plt.legend()
# plt.show()


############################  ANIMATION ############################


x_koordinata = solution[:, 0]
y_koordinata = solution[:, 1]
x_magnets = magnets[:, 0]
y_magnets = magnets[:, 1]

fig = plt.figure()
axis = fig.add_subplot(xlim=(-2, 2), ylim=(-2, 2))
# axis.grid()   ne rabim več, ker uporabljam plt.style.use("bmh")
line, = axis.plot([], [], "o-", linewidth=2)
time_text = axis.text(0.05, 0.9, "", transform=axis.transAxes)
plt.plot(x_magnets, y_magnets, "o", color="black", markersize=5)


def animate(i):
    x = x_koordinata[i]
    y = y_koordinata[i]
    line.set_data(x, y)
    # "time=%.1f" % čas je enako kot da napisemo "time={}s".format(round(i*dt, 1))
    time_text.set_text("time =%.1fs" % (i * 0.02))
    return line, time_text


plt.grid(lw=0.2)
ani = animation.FuncAnimation(
    fig, animate, frames=2000, interval=70, blit=False)

# writervideo = animation.PillowWriter(fps=26)
# ani.save("C:/Users/David/Documents/IPT/animacija.gif", writer=writervideo)
plt.show()

############################  2D plot of field strenght ############################
# def combine_two_arrays(array1, array2):
# 	new = []
# 	for i in range(len(array1)):
# 		new.append([array1[i], array2[i]])
# 	return np.array(new)

# def expand_solution(x, y, solution):
#     N = len(x)
#     M = len(y)


# X, Y = np.meshgrid(x, y)
# # X, Y = np.meshgrid(solution[:, 0], solution[:, 1])

# plt.contourf(X, Y, combine_two_arrays(solution[:, 0], solution[:, 1]), levels=len(X[0])//2, cmap="cividis")
# # plt.contourf(X, Y, position_of_magnets, levels=len(X[0])//2, cmap="cividis")

# cbar = plt.colorbar(orientation="vertical", aspect=10)
# # cbar.set_label(label="Hitrost tekočine v cevi", size = 15)
# plt.xlabel("x", fontsize=12)
# plt.ylabel("y", fontsize=12)
# # plt.title("Hitrostni profil toka tekočine skozi presek cevi", fontweight="bold", fontsize=15)
# # plt.xlim((-1.05, 1.05))
# # plt.ylim((0, 1.05))
# plt.show()


############################  3D plot of strenght of B  ############################
# ax = plt.axes(projection ="3d")
# ax.plot_surface(X, Y, mag_field, cmap ="cividis")  #viridis   #, edgecolor ="green"  To je barva mreže po, ki teče čez graf
# ax.set_xlabel("x", fontsize=12)
# ax.set_ylabel("y", fontsize=12)
# ax.set_zlabel("B (x, y)", fontsize=12)
# ax.set_title("3D representaion of B in xy plain", fontweight="bold", fontsize=15)
# plt.show()
