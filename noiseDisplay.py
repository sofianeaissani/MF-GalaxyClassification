import sys
import matplotlib.pyplot as plt
from libs.pic_process import *
import numpy as np
from scipy import ndimage, misc
import imageio

parameter = 100
coef = 10


### 1st test image : matrix containing maximum values all along the diagonal
"""
a = np.ones((100, 100))*128 + np.eye(100)*128 
b = a.copy()
b = uniform_poisson_noise(b, parameter)
c = a.copy()
c = adaptive_poisson_noise(c, coef)
"""

### 2nd test image : didier (the current directory must be Projet-Galaxies-2020)
"""
a = imageio.imread("osef/didier.png")
a = np.sum(a[:,:,[0, 1, 2]], axis=2)/3
b = a.copy()
b = uniform_poisson_noise(b, parameter)
c = a.copy()
c = adaptive_poisson_noise(c, coef)
"""

### 3rd test image : simulated galaxy (the range of luminosity is way above 256 pixels, it can go up to 10^8, 10^9)

a = charger_fichier_A("DONTPUSH/4_SYNTHESE/output_symm.dat")[0]
print(a)
b = a.copy()
b = uniform_poisson_noise(b, parameter)
c = a.copy()
c = adaptive_poisson_noise(c, coef)

### Plotting

size_window = [8,8]

fig = plt.figure(figsize = (*size_window,))
fig.suptitle("Comparaison des bruits de Poisson")

fig.add_subplot(131)
plt.imshow(a, cmap="viridis")
plt.title("Aucun\ntraitement")

fig.add_subplot(132)
plt.title("Bruit de Poisson uniforme\nde paramètre " + str(parameter))
plt.imshow(b, cmap="viridis")

fig.add_subplot(133)
plt.title("Bruit de Poisson adaptatif\nde coefficient " + str(coef))
plt.imshow(c, cmap="viridis")

plt.show()
