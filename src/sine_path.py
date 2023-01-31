import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(-500, 500, num=1024)
amp = 30
freq = 25

y = amp * np.sin(x / (freq * math.pi)) + amp

for idx in range(len(x)):
    print(f" {x[idx]} {y[idx]}", end="")

# plt.plot(x, y + 75, label="path")
# plt.legend()

# plt.xlim(-500, 500)
# plt.ylim(0, 250)

# plt.show()
