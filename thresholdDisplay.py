import sys
import matplotlib.pyplot as plt
from libs.pic_process import *
import numpy as np
from scipy import ndimage, misc
import imageio

a = get_image("1_TESTS_FANCY/NGC1300_HAWK-I-r+g+b.dat")[0]
NUMAX = np.max(a)
NUMIN = np.min(a)

b = a.copy()
c = a.copy()
d = a.copy()
e = a.copy()

a_ = a >= NUMIN
b = b >= (NUMAX-NUMIN)/4
c = c >= 2*(NUMAX-NUMIN)/4
d = d >= 3*(NUMAX-NUMIN)/4
e = e >= 10000000000000000

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
