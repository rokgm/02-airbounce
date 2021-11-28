import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10, 100)

plt.plot(x, 0.5*x, 'k')
plt.plot(x, -2*x, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.axhline(0,color="k") 
plt.axvline(0,color="k") 
plt.text(-10,-10,[0,0])
plt.text(0.2, 0.04, r'$\alpha$')
plt.text(-0.08, 0.2, r'$\alpha$')
plt.text(0.9, -0.05, 'x')
plt.text(-0.05, 0.9, 'y')
plt.text(0.9, 0.4, 'u')
plt.text(-0.55, 0.9, 'v')
# plt.savefig('kot_frizbija')
plt.show()