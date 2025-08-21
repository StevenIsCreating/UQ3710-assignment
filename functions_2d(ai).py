import numpy as np
import matplotlib.pyplot as plt

# ---- Grid ----
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# ---- Frequency and phase ----
kx, ky = 2.0, 2.0  # adjust for different stripe density
phase = 0.0

# ---- 2D sine / cosine ----
S = np.sin(kx * X + ky * Y + phase)
C = np.cos(kx * X + ky * Y + phase)

# ---- Gaussian window ----
G = np.exp(- (X**2 + Y**2) / 2.0)

# ---- Gabor-like pattern ----
Gabor = G * C

# ---- Plot helper ----
def show_img(arr, title):
    plt.figure()
    plt.imshow(arr, origin='lower')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show all
show_img(S, '2D Sine')
show_img(C, '2D Cosine')
show_img(Gabor, 'Gaussian × Cosine (Gabor-like)')

