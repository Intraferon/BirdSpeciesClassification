import matplotlib.pyplot as plt
import numpy as np

def hann_window():
    plt.xticks(np.linspace(0, 512, 9))
    x_ = np.linspace(0, 512, 512)
    y_ = 0.5 - 0.5 * np.cos((2 * np.pi * x_) / 512)
    return x_, y_

ax = plt.subplot()

x, y = hann_window()

ax.margins(x=0, y=0.01)
ax.plot(x, y, "g")

ax.spines[['right', 'top']].set_visible(False)

plt.show()
