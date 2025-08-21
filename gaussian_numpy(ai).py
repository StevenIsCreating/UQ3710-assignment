# gaussian_numpy.py
import numpy as np
import matplotlib.pyplot as plt

# ---- 2D Gaussian with NumPy ----
# Grid: from -4 to 4 with step 0.01
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# Gaussian: exp(-(x^2 + y^2)/2)
Z = np.exp(- (X**2 + Y**2) / 2.0)

# Plot
plt.imshow(Z, origin='lower')
plt.title('2D Gaussian (NumPy)')
plt.axis('off')
plt.tight_layout()

# show
plt.show()
