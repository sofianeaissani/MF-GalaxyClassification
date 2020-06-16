import sys
import matplotlib.pyplot as plt
from libs.imProcess import *
import numpy as np
from scipy import ndimage, misc
import imageio

a = get_image("data/dataset1_z075-100_M214/0005_149p971429482882_1p6524724303036218_acs_I_mosaic_30mas_sci.fits")[0]
a = second_inflexion_point(a)
NUMAX = np.max(a)
NUMIN = np.min(a)

b = a.copy()
c = a.copy()
d = a.copy()
e = a.copy()

a_ = a >= NUMIN
b = b >= 2*(NUMAX-NUMIN)/6
c = c >= 3*(NUMAX-NUMIN)/6
d = d >= 4*(NUMAX-NUMIN)/6
e = e >= 5*(NUMAX-NUMIN)/6

### Plotting

size_window = [8, 5]

fig = plt.figure(figsize = (*size_window,))

fig.add_subplot(253)
plt.imshow(a, cmap="rainbow")

fig.add_subplot(256)
plt.imshow(a_, cmap="binary")

fig.add_subplot(257)
plt.imshow(b, cmap="binary")

fig.add_subplot(258)
plt.imshow(c, cmap="binary")

fig.add_subplot(259)
plt.imshow(d, cmap="binary")

fig.add_subplot(2,5,10)
plt.imshow(e, cmap="binary")

plt.show()
